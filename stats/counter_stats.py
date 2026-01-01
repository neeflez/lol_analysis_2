import argparse
import csv
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE = "https://www.counterstats.net"
ALL_CHAMPS = f"{BASE}/all-champions"

UA = "Mozilla/5.0 (compatible; CounterStatsScraper/1.0; +https://example.local)"

ROLE_CANON = {
    "Top Lane": "top",
    "Mid Lane": "mid",
    "In the Jungle": "jungle",
    "Support": "support",
    "Bot Lane": "bot",         
    "Bottom Lane": "bot",       
    "ADC": "bot",
}

@dataclass
class ChampionRoleStats:
    champion: str
    champion_slug: str
    role: str
    win_rate: Optional[float]
    pick_rate: Optional[float]

@dataclass
class CounterRow:
    champion: str
    champion_slug: str
    role: str
    counter: str
    counter_rating: Optional[float]
    counter_win_rate: Optional[float]
    matchup_url: str

@dataclass
class MatchupExtra:
    champion: str
    role: str
    counter: str
    matchup_url: str
    games_count: Optional[int]


def safe_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def safe_int(x: str) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None

def get_soup(session: requests.Session, url: str, timeout: int = 30) -> BeautifulSoup:
    r = session.get(url, timeout=timeout)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")

def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def extract_champions_from_all(session: requests.Session) -> List[Tuple[str, str, str]]:
    """
    Returns list of (champion_name, champion_slug, champion_url)
    """
    soup = get_soup(session, ALL_CHAMPS)
    champs = []
    
    for a in soup.select("a[href]"):
        href = a.get("href", "")
        text = normalize_space(a.get_text(" ", strip=True))
        if not href.startswith("/league-of-legends/"):
            continue
        if not text.endswith("Counters"):
            continue
       
        slug = href.split("/")[-1]
        name = text.replace("Counters", "").strip()
        champs.append((name, slug, urljoin(BASE, href)))

  
    seen = set()
    out = []
    for name, slug, url in champs:
        key = (slug.lower(), url)
        if key in seen:
            continue
        seen.add(key)
        out.append((name, slug, url))
    return out

def find_role_blocks(soup: BeautifulSoup) -> List[Tuple[str, BeautifulSoup]]:
    """
    Heuristic: role blocks start around headings like:
      "Hide Top Lane ... Top Lane"
      "Hide Jungle ... In the Jungle"
      "Hide Support ... As a Support"
    In the parsed HTML we can search for H2 tags that contain role text,
    then treat following siblings until next H2 as that role section.
    """
    blocks = []
    h2s = soup.find_all(["h2"])
    for h2 in h2s:
        t = normalize_space(h2.get_text(" ", strip=True))
      
        matched_label = None
        for label in ROLE_CANON.keys():
            if label in t:
                matched_label = label
                break
        if not matched_label:
            continue

 
        section_nodes = []
        node = h2.next_sibling
        while node is not None:
            if getattr(node, "name", None) == "h2":
                break
            section_nodes.append(node)
            node = node.next_sibling

  
        frag_html = "".join(str(n) for n in section_nodes)
        frag = BeautifulSoup(frag_html, "html.parser")
        blocks.append((matched_label, frag))
    return blocks

def extract_win_pick_from_block(role_label: str, block: BeautifulSoup) -> Tuple[Optional[float], Optional[float]]:

    text = normalize_space(block.get_text(" ", strip=True))
    # allow decimals for win rate
    m = re.search(r"(\d+(?:\.\d+)?)%\s*Win Rate\s*(\d+(?:\.\d+)?)%\s*Pick Rate", text, re.IGNORECASE)
    if not m:
        return None, None
    return safe_float(m.group(1)), safe_float(m.group(2))

def extract_counters_from_block(champion: str, champion_slug: str, role_canon: str, block: BeautifulSoup) -> List[CounterRow]:

    counters: List[CounterRow] = []

    heading = None
    for h in block.find_all(["h3", "h4"]):
        if "Best Picks Against" in normalize_space(h.get_text(" ", strip=True)):
            heading = h
            break
    if heading is None:
        return counters


    for a in heading.find_all_next("a", href=True):
       
        if a.find_previous(["h3", "h4"]) != heading:
            break

        txt = normalize_space(a.get_text(" ", strip=True))
        href = a["href"]
        if not href.startswith("/league-of-legends/"):
            continue

        m = re.match(r"(?P<rating>\d+(?:\.\d+)?)\s+(?P<name>.+?)\s+Win\s+(?P<win>\d+(?:\.\d+)?)%", txt)
        if not m:
            continue

        counter_name = m.group("name").strip()
        counter_rating = safe_float(m.group("rating"))
        counter_win = safe_float(m.group("win"))
        matchup_url = urljoin(BASE, href)

        counters.append(
            CounterRow(
                champion=champion,
                champion_slug=champion_slug,
                role=role_canon,
                counter=counter_name,
                counter_rating=counter_rating,
                counter_win_rate=counter_win,
                matchup_url=matchup_url,
            )
        )

    return counters

def extract_worst_matchups_from_block(champion: str, champion_slug: str, role_canon: str, block: BeautifulSoup) -> List[CounterRow]:
    
    worst: List[CounterRow] = []
   
    heading = None
    for h in block.find_all(["h3", "h4"]):
        heading_text = normalize_space(h.get_text(" ", strip=True))
        if "Worst" in heading_text or "Weak" in heading_text:
            heading = h
            break
    
    if heading is None:
        return worst

    for a in heading.find_all_next("a", href=True):
       
        if a.find_previous(["h3", "h4"]) != heading:
            break

        txt = normalize_space(a.get_text(" ", strip=True))
        href = a["href"]
        if not href.startswith("/league-of-legends/"):
            continue

        m = re.match(r"(?P<rating>\d+(?:\.\d+)?)\s+(?P<name>.+?)\s+Win\s+(?P<win>\d+(?:\.\d+)?)%", txt)
        if not m:
         
            m2 = re.match(r"(?P<name>.+?)\s+Win\s+(?P<win>\d+(?:\.\d+)?)%", txt)
            if m2:
                counter_name = m2.group("name").strip()
                counter_rating = None  
                counter_win = safe_float(m2.group("win"))
            else:
                continue
        else:
            counter_name = m.group("name").strip()
            counter_rating = safe_float(m.group("rating"))
            counter_win = safe_float(m.group("win"))
        
        matchup_url = urljoin(BASE, href)

        worst.append(
            CounterRow(
                champion=champion,
                champion_slug=champion_slug,
                role=role_canon,
                counter=counter_name,
                counter_rating=counter_rating,
                counter_win_rate=counter_win,
                matchup_url=matchup_url,
            )
        )

    return worst

def extract_games_count_from_matchup(soup: BeautifulSoup) -> Optional[int]:
   
    text = normalize_space(soup.get_text(" ", strip=True))
    m = re.search(r"\b(\d{1,7})\s+VS\s+in\s+[A-Za-z ]+", text, re.IGNORECASE)
    if not m:
        return None
    return safe_int(m.group(1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out", help="Output directory for CSVs")
    ap.add_argument("--sleep", type=float, default=0.8, help="Delay between HTTP requests (seconds)")
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--limit_champions", type=int, default=0, help="0 = no limit (debug option)")
    ap.add_argument("--include_matchups", action="store_true", help="Fetch matchup pages to extract extra fields (heavier)")
    ap.add_argument("--max_matchups_per_role", type=int, default=0, help="0 = all counters; otherwise limit per champion-role")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": UA})

    champions = extract_champions_from_all(session)
    if args.limit_champions and args.limit_champions > 0:
        champions = champions[: args.limit_champions]

    role_rows: List[ChampionRoleStats] = []
    counter_rows: List[CounterRow] = []
    matchup_extras: List[MatchupExtra] = []

    for idx, (champ_name, champ_slug, champ_url) in enumerate(champions, start=1):
        print(f"[{idx}/{len(champions)}] {champ_name} -> {champ_url}")
        try:
            soup = get_soup(session, champ_url, timeout=args.timeout)
        except Exception as e:
            print(f"  ! failed champion page: {e}")
            continue
        time.sleep(args.sleep)

        blocks = find_role_blocks(soup)
    
        if not blocks:
            blocks = [("Unknown", soup)]

        for role_label, block in blocks:
            role_canon = ROLE_CANON.get(role_label, normalize_space(role_label).lower().replace(" ", "_"))

            win_rate, pick_rate = extract_win_pick_from_block(role_label, block)
            role_rows.append(
                ChampionRoleStats(
                    champion=champ_name,
                    champion_slug=champ_slug,
                    role=role_canon,
                    win_rate=win_rate,
                    pick_rate=pick_rate,
                )
            )

            counters = extract_counters_from_block(champ_name, champ_slug, role_canon, block)
            if args.max_matchups_per_role and args.max_matchups_per_role > 0:
                counters = counters[: args.max_matchups_per_role]
            counter_rows.extend(counters)
            
   
            worst_matchups = extract_worst_matchups_from_block(champ_name, champ_slug, role_canon, block)
            if args.max_matchups_per_role and args.max_matchups_per_role > 0:
                worst_matchups = worst_matchups[: args.max_matchups_per_role]
            counter_rows.extend(worst_matchups)

            if args.include_matchups:
                for c in counters:
                    try:
                        msoup = get_soup(session, c.matchup_url, timeout=args.timeout)
                        games = extract_games_count_from_matchup(msoup)
                        matchup_extras.append(
                            MatchupExtra(
                                champion=c.champion,
                                role=c.role,
                                counter=c.counter,
                                matchup_url=c.matchup_url,
                                games_count=games,
                            )
                        )
                    except Exception as e:
                        print(f"    ! failed matchup: {c.matchup_url} ({e})")
                    time.sleep(args.sleep)


    roles_path = os.path.join(args.outdir, "champion_roles.csv")
    with open(roles_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["champion", "champion_slug", "role", "win_rate_pct", "pick_rate_pct"])
        for r in role_rows:
            w.writerow([r.champion, r.champion_slug, r.role, r.win_rate, r.pick_rate])

    counters_path = os.path.join(args.outdir, "counters.csv")
    with open(counters_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "champion", "champion_slug", "role",
            "counter", "counter_rating",
            "counter_win_rate_pct",
            "matchup_url"
        ])
        for c in counter_rows:
            w.writerow([
                c.champion, c.champion_slug, c.role,
                c.counter, c.counter_rating,
                c.counter_win_rate,
                c.matchup_url
            ])

    if args.include_matchups:
        extra_path = os.path.join(args.outdir, "matchups_extra.csv")
        with open(extra_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["champion", "role", "counter", "matchup_url", "games_count"])
            for m in matchup_extras:
                w.writerow([m.champion, m.role, m.counter, m.matchup_url, m.games_count])

    print("\nDONE")
    print(f"  - {roles_path}")
    print(f"  - {counters_path}")
    if args.include_matchups:
        print(f"  - {os.path.join(args.outdir, 'matchups_extra.csv')}")


if __name__ == "__main__":
    main()
