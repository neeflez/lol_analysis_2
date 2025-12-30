
import os
import requests
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("RIOT_API_KEY")
HEADERS = {"X-Riot-Token": API_KEY}

LOG_DIR = "data/logs"
RAW_DIR = "data/raw"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)


def riot_get(name: str, url: str):
    """Wysyła zapytanie GET do Riot API i zapisuje wynik"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        response = requests.get(url, headers=HEADERS)
        status = response.status_code

        log_entry = {
            "name": name,
            "url": url,
            "status": status,
            "ok": response.ok,
            "timestamp": timestamp
        }

        # zapis loga
        with open(f"{LOG_DIR}/{name}_{timestamp}.json", "w") as f:
            json.dump(log_entry, f, indent=2)

        # zapis danych jeśli OK
        if response.ok:
            with open(f"{RAW_DIR}/{name}_{timestamp}.json", "w") as f:
                json.dump(response.json(), f, indent=2)

        print(f"[{status}] {name}")
        return response.json() if response.ok else None

    except Exception as e:
        print(f"[ERROR] {name}: {e}")
        return None
