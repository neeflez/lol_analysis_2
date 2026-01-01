# Quick Start Guide

## 1. Setup (5 minut)

```bash
# Zainstaluj zależności
pip install -r requirements.txt

# Skopiuj i skonfiguruj .env
cp .env.example .env  
# Edytuj .env i wpisz swój RIOT_API_KEY

# Test konfiguracji
python test_setup.py
```

## 2. Pierwsze uruchomienie (10 graczy demo)

```bash
python analysis/main.py --num_players 10 --out data/output/demo.csv
```

Czas: ~2-3 minuty

## 3. Sprawdź wynik

```python
import pandas as pd
df = pd.read_csv("data/output/demo.csv")
print(df.head())
print(f"Liczba rekordów: {len(df)}")
print(f"Balance win: {df['win'].value_counts()}")
```

## 4. Pełny dataset (2000 graczy)

```bash
python analysis/main.py --num_players 2000 --out data/output/gold_full.csv
```

⚠️ Czas: ~1-2 godziny (rate limiting)

Pipeline zapisuje postęp - możesz przerwać i wznowić.

## 5. Inne konfiguracje

```bash
# EUW Platinum
python analysis/main.py --num_players 500 --platform EUW1 --tier PLATINUM

# 3 mecze na gracza
python analysis/main.py --num_players 100 --matches_per_player 3

# Świeże dane (clear cache)
python analysis/main.py --num_players 100 --clear_cache
```

## Troubleshooting

- Test nie przechodzi? → Sprawdź `.env` i RIOT_API_KEY
- Rate limit 429? → Pipeline czeka automatycznie
- Błąd "No module"? → Uruchamiaj z głównego katalogu

Więcej: [README.md](README.md)
