"""
Riot API Endpoints Configuration

Platform Routing (dla platform-specific calls):
- EUW1, EUNE (EUN1), NA1, KR, etc.

Regional Routing (dla match/account data):
- EUROPE, AMERICAS, ASIA, SEA
"""

# === REGIONAL ENDPOINTS (match, account) ===
REGIONAL_BASE = {
    "EUROPE": "https://europe.api.riotgames.com",
    "AMERICAS": "https://americas.api.riotgames.com",
    "ASIA": "https://asia.api.riotgames.com"
}

# === PLATFORM ENDPOINTS (summoner, league) ===
PLATFORM_BASE = {
    "EUW1": "https://euw1.api.riotgames.com",
    "EUN1": "https://eun1.api.riotgames.com",
    "NA1": "https://na1.api.riotgames.com",
    "KR": "https://kr.api.riotgames.com"
}


def get_league_entries_url(platform: str = "EUN1", tier: str = "GOLD", division: str = "I", page: int = 1):
    """
    Pobiera URL do League Entries (ranked players).
    
    Args:
        platform: EUW1, EUN1, NA1, etc.
        tier: IRON, BRONZE, SILVER, GOLD, PLATINUM, DIAMOND, MASTER, GRANDMASTER, CHALLENGER
        division: I, II, III, IV (dla tier poniżej MASTER)
        page: Numer strony (każda strona max 205 graczy)
    """
    base = PLATFORM_BASE.get(platform, PLATFORM_BASE["EUN1"])
    return f"{base}/lol/league/v4/entries/RANKED_SOLO_5x5/{tier}/{division}?page={page}"


def get_summoner_by_id_url(platform: str, summoner_id: str):
    """Summoner by encrypted summoner ID"""
    base = PLATFORM_BASE.get(platform, PLATFORM_BASE["EUN1"])
    return f"{base}/lol/summoner/v4/summoners/{summoner_id}"


def get_summoner_by_name_url(platform: str, summoner_name: str):
    """Summoner by summoner name"""
    base = PLATFORM_BASE.get(platform, PLATFORM_BASE["EUN1"])
    return f"{base}/lol/summoner/v4/summoners/by-name/{summoner_name}"


def get_match_ids_by_puuid_url(region: str, puuid: str, start: int = 0, count: int = 20, queue: int = None):
    """
    Lista match IDs dla gracza.
    
    Args:
        region: EUROPE, AMERICAS, ASIA
        puuid: Player UUID
        start: Offset
        count: Max 100
        queue: Queue ID (420 = Ranked Solo, 440 = Ranked Flex, etc.)
    """
    base = REGIONAL_BASE.get(region, REGIONAL_BASE["EUROPE"])
    url = f"{base}/lol/match/v5/matches/by-puuid/{puuid}/ids?start={start}&count={count}"
    if queue:
        url += f"&queue={queue}"
    return url


def get_match_details_url(region: str, match_id: str):
    """Match details (summary)"""
    base = REGIONAL_BASE.get(region, REGIONAL_BASE["EUROPE"])
    return f"{base}/lol/match/v5/matches/{match_id}"


def get_match_timeline_url(region: str, match_id: str):
    """Match timeline"""
    base = REGIONAL_BASE.get(region, REGIONAL_BASE["EUROPE"])
    return f"{base}/lol/match/v5/matches/{match_id}/timeline"


# === LEGACY ENDPOINTS (kompatybilność) ===
ENDPOINTS = {
    "account_by_riot_id": "https://europe.api.riotgames.com/riot/account/v1/accounts/by-riot-id/maciek2016pl2/mciek",
    "match_ids": "https://europe.api.riotgames.com/lol/match/v5/matches/by-puuid/PUUID_TUTAJ/ids?start=0&count=10",
    "match_details": "https://europe.api.riotgames.com/lol/match/v5/matches/EUN1_3873475397"
}

