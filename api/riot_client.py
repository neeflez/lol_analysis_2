
import os
import requests
import json
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("RIOT_API_KEY")
HEADERS = {"X-Riot-Token": API_KEY}

LOG_DIR = "data/logs"
RAW_DIR = "data/raw"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)


def riot_get(name: str, url: str, max_retries: int = 3, save_raw: bool = False):
    """
    Wysyła zapytanie GET do Riot API z retry logic i rate limiting.
    
    Args:
        name: Nazwa zapytania (do logowania)
        url: URL endpointu
        max_retries: Maksymalna liczba prób
        save_raw: Czy zapisywać surowe dane do pliku (domyślnie False dla pipeline)
    
    Returns:
        JSON response lub None w przypadku błędu
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=30)
            status = response.status_code
            
            # Sukces
            if response.ok:
                if save_raw:
                    with open(f"{RAW_DIR}/{name}_{timestamp}.json", "w") as f:
                        json.dump(response.json(), f, indent=2)
                return response.json()
            
            # Rate limit - czekamy i próbujemy ponownie
            elif status == 429:
                retry_after = int(response.headers.get("Retry-After", 10))
                print(f"[429] Rate limit dla {name}, czekam {retry_after}s...")
                time.sleep(retry_after)
                continue
            
            # Błędy serwera - retry z backoff
            elif status >= 500:
                wait = 2 ** attempt
                print(f"[{status}] Błąd serwera dla {name}, retry {attempt+1}/{max_retries} po {wait}s")
                time.sleep(wait)
                continue
            
            # Inne błędy (404, 403 etc.) - nie retryujemy
            else:
                print(f"[{status}] Błąd dla {name}: {url}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"[TIMEOUT] {name}, próba {attempt+1}/{max_retries}")
            time.sleep(2 ** attempt)
            continue
            
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return None
    
    print(f"[FAILED] {name} po {max_retries} próbach")
    return None


def riot_get_with_log(name: str, url: str):
    """Wersja z pełnym logowaniem (kompatybilność wsteczna)"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = riot_get(name, url, save_raw=True)
    
    log_entry = {
        "name": name,
        "url": url,
        "status": 200 if result else "failed",
        "ok": result is not None,
        "timestamp": timestamp
    }
    
    with open(f"{LOG_DIR}/{name}_{timestamp}.json", "w") as f:
        json.dump(log_entry, f, indent=2)
    
    return result
