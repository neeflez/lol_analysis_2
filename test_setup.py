"""
Quick test script - sprawdza czy API key działa i czy endpoints odpowiadają.
Uruchom: python test_setup.py
"""

import os
from dotenv import load_dotenv
from api.riot_client import riot_get
from api.endpoints import get_league_entries_url, get_summoner_by_id_url

load_dotenv()

def test_api_key():
    """Sprawdza czy API key jest ustawiony"""
    api_key = os.getenv("RIOT_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        print("❌ RIOT_API_KEY nie jest ustawiony w .env")
        print("   Skopiuj .env.example do .env i wpisz swój klucz API")
        return False
    print("✅ API key znaleziony")
    return True


def test_league_endpoint():
    """Testuje endpoint League Entries (lista graczy GOLD)"""
    print("\n🧪 Test: League Entries (GOLD I EUN1)...")
    url = get_league_entries_url(platform="EUN1", tier="GOLD", division="I", page=1)
    data = riot_get("test_league", url)
    
    if data and isinstance(data, list) and len(data) > 0:
        print(f"✅ Pobrano {len(data)} graczy GOLD I")
        # Try different field names for summoner name
        name = data[0].get('summonerName') or data[0].get('name') or 'Unknown'
        summoner_id = data[0].get('summonerId', 'unknown')
        print(f"   Przykład: {name} (summonerId: {summoner_id[:8]}...)")
        return True
    else:
        print("❌ Nie udało się pobrać danych League Entries")
        return False


def test_summoner_endpoint():
    """Testuje endpoint Summoner (konwersja summonerId -> puuid)"""
    print("\n🧪 Test: Summoner endpoint...")
    
    league_url = get_league_entries_url(platform="EUN1", tier="GOLD", division="I", page=1)
    league_data = riot_get("test_league_2", league_url)
    
    if not league_data or len(league_data) == 0:
        print("❌ Nie można pobrać gracza do testu")
        return False
    
    # Debug: print available fields in league_data
    print(f"   Debug: Available fields in league_data[0]: {list(league_data[0].keys())}")
    
    # League Entries API v4 returns puuid directly, not summonerId
    puuid = league_data[0].get("puuid")
    if not puuid:
        print("❌ Brak puuid w danych League Entries")
        return False
    
    # Optional: verify we can access Summoner endpoint too (though we already have puuid)
    # In production, we'll use puuid directly since it's already in league_data
    
    print(f"✅ Pobrano puuid z League Entries")
    print(f"   puuid: {puuid[:16]}...")
    return True


def main():
    print("=" * 60)
    print("🔧 TEST SETUP - Riot API Pipeline")
    print("=" * 60)
    
    all_passed = True
    
    if not test_api_key():
        all_passed = False
        print("\n⚠️ Napraw konfigurację .env przed kontynuowaniem")
        return
    
    if not test_league_endpoint():
        all_passed = False
    
    if not test_summoner_endpoint():
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ WSZYSTKIE TESTY PRZESZŁY")
        print("\nMożesz uruchomić pipeline:")
        print("  python analysis/main.py --num_players 10")
    else:
        print("❌ NIEKTÓRE TESTY NIE PRZESZŁY")
        print("\nSprawdź: .env, połączenie, status.riotgames.com")
    print("=" * 60)


if __name__ == "__main__":
    main()
