"""
Storage module for caching and checkpointing pipeline data.
Zapisuje pośrednie wyniki w data/cache/ w formacie JSON.
"""

import json
import os
from typing import Any, Dict, List, Optional


CACHE_DIR = "data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def save_checkpoint(name: str, data: Any) -> None:
    """Zapisuje checkpoint do pliku JSON"""
    filepath = os.path.join(CACHE_DIR, f"{name}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[OK] Checkpoint saved: {filepath}")


def load_checkpoint(name: str) -> Optional[Any]:
    """Wczytuje checkpoint z pliku JSON. Zwraca None jeśli nie istnieje."""
    filepath = os.path.join(CACHE_DIR, f"{name}.json")
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[OK] Checkpoint loaded: {filepath}")
    return data


def checkpoint_exists(name: str) -> bool:
    """Sprawdza czy checkpoint istnieje"""
    filepath = os.path.join(CACHE_DIR, f"{name}.json")
    return os.path.exists(filepath)


def clear_cache() -> None:
    """Usuwa wszystkie pliki cache"""
    for filename in os.listdir(CACHE_DIR):
        filepath = os.path.join(CACHE_DIR, filename)
        if os.path.isfile(filepath):
            os.remove(filepath)
    print("[OK] Cache cleared")
