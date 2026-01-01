"""
Pipeline do pobierania danych z Riot API dla graczy GOLD.

Sekwencja:
1. Pobierz listę GOLD players (zawiera już puuid z League Entries API v4)
2. Dla każdego pobierz listę match_ids
3. Dla wybranych match_ids pobierz timeline i oblicz features@15
4. Pobierz match summary i określ win/lose
5. Zapisz do CSV
"""

import csv
import time
import os
from typing import Dict, List, Optional, Tuple

from api.riot_client import riot_get
from api.endpoints import (
    get_league_entries_url,
    get_match_ids_by_puuid_url,
    get_match_details_url,
    get_match_timeline_url
)
from api.storage import save_checkpoint, load_checkpoint
from features.extract import extract_features


def get_gold_players(
    num_players: int,
    platform: str = "EUN1",
    tier: str = "GOLD",
    division: str = "I"
) -> List[Dict]:
    """
    Pobiera listę graczy GOLD z League Entries API.
    League Entries v4 API zwraca puuid bezpośrednio.
    
    Returns:
        List[Dict] z kluczami: puuid, tier, rank, leaguePoints
    """
    checkpoint_name = f"gold_players_{platform}_{tier}_{division}"
    cached = load_checkpoint(checkpoint_name)
    if cached and len(cached) >= num_players:
        print(f"[CACHE] Using cache for gold_players ({len(cached)} players)")
        return cached[:num_players]
    
    players = []
    page = 1
    max_pages = 50  # safety limit
    
    print(f"[*] Fetching {tier} {division} players from {platform}...")
    
    while len(players) < num_players and page <= max_pages:
        url = get_league_entries_url(platform, tier, division, page)
        data = riot_get(f"league_page_{page}", url)
        
        if not data or len(data) == 0:
            print(f"  No more data on page {page}")
            break
        
        for entry in data:
            # League Entries API v4 zwraca puuid
            players.append({
                "puuid": entry["puuid"],
                "tier": entry.get("tier"),
                "rank": entry.get("rank"),
                "leaguePoints": entry.get("leaguePoints")
            })
            
            if len(players) >= num_players:
                break
        
        print(f"  Page {page}: collected {len(players)}/{num_players} players")
        page += 1
        time.sleep(0.5)  # rate limiting
    
    save_checkpoint(checkpoint_name, players)
    return players[:num_players]


def get_match_ids(
    puuids: List[str],
    matches_per_player: int = 1,
    region: str = "EUROPE",
    queue: Optional[int] = 420  # 420 = Ranked Solo
) -> Dict[str, List[str]]:
    """
    Pobiera match_ids dla każdego puuid.
    
    Returns:
        Dict[puuid -> List[match_id]]
    """
    checkpoint_name = f"match_ids_{region}"
    cached = load_checkpoint(checkpoint_name)
    if cached:
        print(f"[CACHE] Using cache for match_ids ({len(cached)} players)")
        return cached
    
    match_map = {}
    total = len(puuids)
    
    print(f"[*] Fetching match_ids for {total} players...")
    
    for idx, puuid in enumerate(puuids, 1):
        url = get_match_ids_by_puuid_url(
            region, puuid,
            start=0,
            count=matches_per_player * 2,
            queue=queue
        )
        data = riot_get(f"matches_{puuid[:8]}", url)
        
        if data and isinstance(data, list) and len(data) > 0:
            match_map[puuid] = data[:matches_per_player]
        else:
            print(f"  [!] No matches for puuid {puuid[:8]}")
        
        if idx % 10 == 0:
            print(f"  Progress: {idx}/{total}")
            time.sleep(0.5)
    
    save_checkpoint(checkpoint_name, match_map)
    print(f"[OK] Fetched matches for {len(match_map)} players")
    return match_map


def get_timeline_features_15(
    match_ids: List[str],
    region: str = "EUROPE"
) -> Dict[str, Dict]:
    """
    Pobiera timeline i oblicza features@15 dla każdego match_id.
    
    Returns:
        Dict[match_id -> features]
    """
    checkpoint_name = f"timeline_features_{region}"
    cached = load_checkpoint(checkpoint_name)
    if cached:
        print(f"[CACHE] Using cache for timeline_features ({len(cached)} matches)")
        return cached
    
    features_map = {}
    total = len(match_ids)
    
    print(f"[*] Fetching timeline for {total} matches...")
    
    for idx, match_id in enumerate(match_ids, 1):
        url = get_match_timeline_url(region, match_id)
        timeline = riot_get(f"timeline_{match_id}", url)
        
        if timeline:
            try:
                features = extract_features(timeline, minute=15)
                features_map[match_id] = features
            except Exception as e:
                print(f"  [!] Error extracting features for {match_id}: {e}")
        else:
            print(f"  [!] No timeline for {match_id}")
        
        if idx % 5 == 0:
            print(f"  Progress: {idx}/{total}")
            time.sleep(0.5)
    
    save_checkpoint(checkpoint_name, features_map)
    print(f"[OK] Computed features for {len(features_map)} matches")
    return features_map


def get_match_outcomes(
    match_ids: List[str],
    puuid_map: Dict[str, List[str]],  # puuid -> [match_ids]
    region: str = "EUROPE"
) -> Dict[Tuple[str, str], int]:
    """
    Pobiera match summary i określa win/lose dla każdego (puuid, match_id).
    
    Returns:
        Dict[(puuid, match_id) -> win (1/0)]
    """
    checkpoint_name = f"match_outcomes_{region}"
    cached = load_checkpoint(checkpoint_name)
    if cached:
        # Convert keys back from string to tuple
        cached_tuples = {tuple(k.split("|")): v for k, v in cached.items()}
        print(f"[CACHE] Using cache for match_outcomes ({len(cached_tuples)} records)")
        return cached_tuples
    
    outcomes = {}
    total = len(match_ids)
    
    # Odwróć mapę: match_id -> list of puuids
    match_to_puuids = {}
    for puuid, matches in puuid_map.items():
        for match_id in matches:
            if match_id not in match_to_puuids:
                match_to_puuids[match_id] = []
            match_to_puuids[match_id].append(puuid)
    
    print(f"[*] Fetching match outcomes for {total} matches...")
    
    for idx, match_id in enumerate(match_ids, 1):
        url = get_match_details_url(region, match_id)
        match_data = riot_get(f"match_{match_id}", url)
        
        if not match_data or "info" not in match_data:
            print(f"  [!] No match data for {match_id}")
            continue
        
        # Mapowanie puuid -> win
        participants = match_data["info"]["participants"]
        for participant in participants:
            p_puuid = participant["puuid"]
            win = 1 if participant["win"] else 0
            
            # Zapisz tylko dla puuidów z naszej listy
            if p_puuid in match_to_puuids.get(match_id, []):
                outcomes[(p_puuid, match_id)] = win
        
        if idx % 5 == 0:
            print(f"  Progress: {idx}/{total}")
            time.sleep(0.5)
    
    # Save with string keys for JSON compatibility
    outcomes_str = {f"{k[0]}|{k[1]}": v for k, v in outcomes.items()}
    save_checkpoint(checkpoint_name, outcomes_str)
    print(f"[OK] Fetched outcomes for {len(outcomes)} records")
    return outcomes


def run_pipeline(
    num_players: int = 100,
    matches_per_player: int = 1,
    platform: str = "EUN1",
    region: str = "EUROPE",
    tier: str = "GOLD",
    division: str = "I",
    output_path: str = "data/output/gold_dataset.csv"
) -> None:
    """
    Główna funkcja pipeline.
    
    Args:
        num_players: Liczba graczy do pobrania
        matches_per_player: Liczba meczów na gracza
        platform: Platform routing (EUN1, EUW1, etc.)
        region: Regional routing (EUROPE, AMERICAS, etc.)
        tier: Tier ranked (GOLD, SILVER, etc.)
        division: Division (I, II, III, IV)
        output_path: Ścieżka do pliku CSV
    """
    print("=" * 70)
    print("[*] RIOT API DATA PIPELINE")
    print("=" * 70)
    print(f"Parameters:")
    print(f"  - Players: {num_players}")
    print(f"  - Matches per player: {matches_per_player}")
    print(f"  - Platform: {platform}")
    print(f"  - Region: {region}")
    print(f"  - Tier: {tier} {division}")
    print(f"  - Output: {output_path}")
    print("=" * 70)
    
    # 1. Pobierz graczy GOLD (zawiera już puuid)
    players = get_gold_players(num_players, platform, tier, division)
    puuids = [p["puuid"] for p in players]
    print(f"\n[OK] Step 1/4: Fetched {len(players)} players\n")
    
    # 2. Pobierz match_ids
    match_map = get_match_ids(puuids, matches_per_player, region)
    
    # Flatten do listy wszystkich match_ids
    all_match_ids = []
    for puuid, matches in match_map.items():
        all_match_ids.extend(matches)
    all_match_ids = list(set(all_match_ids))  # deduplikacja
    print(f"\n[OK] Step 2/4: Fetched {len(all_match_ids)} unique matches\n")
    
    # 3. Pobierz timeline features@15
    features_map = get_timeline_features_15(all_match_ids, region)
    print(f"\n[OK] Step 3/4: Computed features for {len(features_map)} matches\n")
    
    # 4. Pobierz match outcomes
    outcomes_map = get_match_outcomes(all_match_ids, match_map, region)
    print(f"\n[OK] Step 4/4: Fetched outcomes for {len(outcomes_map)} records\n")
    
    # 5. Buduj dataset
    print("[*] Building final dataset...")
    dataset = []
    
    # Utwórz mapping puuid -> player
    puuid_to_player = {p["puuid"]: p for p in players}
    
    for (puuid, match_id), win in outcomes_map.items():
        if match_id not in features_map:
            continue
        
        player = puuid_to_player.get(puuid)
        if not player:
            continue
        
        features = features_map[match_id].copy()
        
        row = {
            "puuid": puuid,
            "matchId": match_id,
            "win": win,
            **features
        }
        dataset.append(row)
    
    print(f"[OK] Dataset contains {len(dataset)} records")
    
    # 6. Zapisz do CSV
    if dataset:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        fieldnames = ["puuid", "matchId", "win"] + list(features_map[all_match_ids[0]].keys())
        
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(dataset)
        
        print(f"\n[OK] Dataset saved: {output_path}")
        print(f"     Records: {len(dataset)}")
        print(f"     Columns: {len(fieldnames)}")
    else:
        print("\n[!] No data to save")
    
    print("=" * 70)
    print("[OK] PIPELINE COMPLETE")
    print("=" * 70)
