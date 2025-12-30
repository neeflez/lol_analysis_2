from api.riot_client import riot_get

def get_timeline(match_id: str):
    url = f"https://europe.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline"
    return riot_get(f"timeline_{match_id}", url)

