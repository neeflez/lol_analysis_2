import csv
import streamlit as st
from api.riot_client import riot_get
from api.endpoints import ENDPOINTS
from api.timeline import get_timeline
from features.extract import extract_features

#for name, url in ENDPOINTS.items():
#    riot_get(name, url)
match_ids = [
    "EUN1_3873475397",
    "EUN1_3873475400",
    "EUN1_3873475401",
    # dodaj kolejne
]

def main():
    dataset = []

    for match_id in match_ids:
        # 1️⃣ Pobranie timeline
        timeline = get_timeline(match_id)
        if not timeline:
            print(f"Nie udało się pobrać timeline dla {match_id}")
            continue

        # 2️⃣ Ekstrakcja cech do 15 minuty
        features = extract_features(timeline, minute=15)

        # 3️⃣ Pobranie wyniku meczu (target)
        match_summary = riot_get(f"summary_{match_id}",
                                 f"https://europe.api.riotgames.com/lol/match/v5/matches/{match_id}")
        if not match_summary:
            print(f"Nie udało się pobrać summary dla {match_id}")
            continue

        # win dla team100 (1 = wygrana, 0 = przegrana)
        team100_win = 1 if match_summary["info"]["teams"][0]["win"] else 0
        features["win"] = team100_win

        dataset.append(features)
        print(f"[OK] Dodano cechy dla meczu {match_id}")

    # 4️⃣ Zapis do CSV
    if dataset:
        keys = dataset[0].keys()
        with open("data/processed/dataset.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(dataset)
        print(f"Dataset zapisany do data/processed/dataset.csv")
    else:
        print("Brak danych do zapisania")

if __name__ == "__main__":
    main()
