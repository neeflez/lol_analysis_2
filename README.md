# Riot API Data Pipeline - League of Legends

Pipeline do automatycznego pobierania i przetwarzania danych graczy z Riot Games API. Zbiera dane graczy z rankingów (np. GOLD), pobiera ich mecze, analizuje timeline @15 minut i tworzy gotowy dataset do analizy ML.

## 📋 Wymagania

- Python 3.8+
- Riot Games API Key (https://developer.riotgames.com/)

## 🚀 Instalacja

```bash
cd lol_analysis_2
pip install -r requirements.txt
```

Skonfiguruj API Key:
```bash
cp .env.example .env
# Edytuj .env i wpisz: RIOT_API_KEY=twoj_klucz_tutaj
```

## 💻 Użycie

### Podstawowe (100 graczy GOLD I)
```bash
python analysis/main.py --num_players 100
```

### Pełny dataset (2000 graczy)
```bash
python analysis/main.py --num_players 2000 --out data/output/full_gold.csv
```

### Inne konfiguracje
```bash
# EUW Platinum
python analysis/main.py --num_players 500 --platform EUW1 --tier PLATINUM --division III

# 3 mecze na gracza
python analysis/main.py --num_players 100 --matches_per_player 3

# Świeże dane (clear cache)
python analysis/main.py --num_players 100 --clear_cache
```

## 📊 Parametry

| Parametr | Domyślnie | Opis |
|----------|-----------|------|
| `--num_players` | 100 | Liczba graczy |
| `--matches_per_player` | 1 | Mecze na gracza |
| `--platform` | EUN1 | EUN1, EUW1, NA1, KR |
| `--region` | EUROPE | EUROPE, AMERICAS, ASIA |
| `--tier` | GOLD | IRON-DIAMOND |
| `--division` | I | I, II, III, IV |
| `--out` | data/output/gold_dataset.csv | Ścieżka CSV |
| `--clear_cache` | - | Wyczyść cache |

## 🔄 Pipeline (5 kroków)

1. **get_gold_players()** → Pobiera graczy z League Entries
2. **get_puuids()** → Konwertuje summonerId → puuid
3. **get_match_ids()** → Pobiera match history
4. **get_timeline_features_15()** → Oblicza features @15 min
5. **get_match_outcomes()** → Określa win/lose

Cache automatyczny w `data/cache/` - możliwe wznawianie po Ctrl+C.

## 📈 Output CSV

Kolumny: `summonerId, puuid, matchId, win` + 22 features:
- `gold_diff`, `cs_diff`, `level_diff`, `xp_diff`
- `kills_diff`, `deaths_diff`, `towers_diff`, `dragons_diff`
- `first_blood_diff`, `first_tower_diff`, etc.

Wszystkie `*_diff` = team100 - team200

## ⏱️ Czas wykonania

- 10 graczy: ~2-3 min
- 100 graczy: ~15-20 min
- 2000 graczy: ~4-6 godz

## 🔧 Rate Limiting

- Auto-retry dla 429 (rate limit)
- Exponential backoff dla 5xx
- Retry-After header handling

## 📝 Więcej info

Zobacz: `QUICKSTART.md`, `ARCHITECTURE.md`

---
**Projekt UM, Styczeń 2026**
