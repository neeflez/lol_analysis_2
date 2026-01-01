"""
Main entry point for Riot API Data Pipeline.

Usage:
    python analysis/main.py --num_players 100 --matches_per_player 1 --out data/output/gold_dataset.csv
    
    # Z cache (wznowienie):
    python analysis/main.py --num_players 2000 --out data/output/full_dataset.csv
    
    # Różne parametry:
    python analysis/main.py --num_players 500 --platform EUW1 --tier PLATINUM --division II
"""

import argparse
import sys
import os

# Dodaj parent directory do path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Riot API Data Pipeline - pobieranie danych graczy GOLD/PLAT/etc.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przykłady użycia:
  # Podstawowe uruchomienie (100 graczy GOLD I z EUN1):
  python analysis/main.py --num_players 100
  
  # Pełny dataset (2000 graczy):
  python analysis/main.py --num_players 2000 --out data/output/full_gold.csv
  
  # Platyna z EUW:
  python analysis/main.py --num_players 500 --platform EUW1 --tier PLATINUM --division III
  
  # Więcej meczów na gracza:
  python analysis/main.py --num_players 100 --matches_per_player 3

Pipeline używa cache - jeśli przerwiesz, uruchom ponownie z tymi samymi parametrami.
Cache znajduje się w data/cache/ i może być usunięty jeśli potrzebujesz świeżych danych.
        """
    )
    
    parser.add_argument(
        "--num_players",
        type=int,
        default=100,
        help="Liczba graczy do pobrania (domyślnie: 100)"
    )
    
    parser.add_argument(
        "--matches_per_player",
        type=int,
        default=1,
        help="Liczba meczów na gracza (domyślnie: 1)"
    )
    
    parser.add_argument(
        "--platform",
        type=str,
        default="EUN1",
        choices=["EUN1", "EUW1", "NA1", "KR"],
        help="Platform routing dla summoner/league endpoints (domyślnie: EUN1)"
    )
    
    parser.add_argument(
        "--region",
        type=str,
        default="EUROPE",
        choices=["EUROPE", "AMERICAS", "ASIA"],
        help="Regional routing dla match endpoints (domyślnie: EUROPE)"
    )
    
    parser.add_argument(
        "--tier",
        type=str,
        default="GOLD",
        choices=["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "DIAMOND"],
        help="Tier ranked (domyślnie: GOLD)"
    )
    
    parser.add_argument(
        "--division",
        type=str,
        default="I",
        choices=["I", "II", "III", "IV"],
        help="Division (I-IV, domyślnie: I)"
    )
    
    parser.add_argument(
        "--out",
        type=str,
        default="data/output/gold_dataset.csv",
        help="Ścieżka do pliku CSV wyjściowego (domyślnie: data/output/gold_dataset.csv)"
    )
    
    parser.add_argument(
        "--clear_cache",
        action="store_true",
        help="Wyczyść cache przed uruchomieniem (świeże dane)"
    )
    
    args = parser.parse_args()
    
    # Clear cache if requested
    if args.clear_cache:
        from api.storage import clear_cache
        clear_cache()
    
    # Uruchom pipeline
    try:
        run_pipeline(
            num_players=args.num_players,
            matches_per_player=args.matches_per_player,
            platform=args.platform,
            region=args.region,
            tier=args.tier,
            division=args.division,
            output_path=args.out
        )
    except KeyboardInterrupt:
        print("\n\n[!] Pipeline interrupted by user")
        print("[*] Progress was saved in cache. Run again to continue.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
