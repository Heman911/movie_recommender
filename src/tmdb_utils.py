# src/tmdb_utils.py
import requests
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "ml-latest-small"
LINKS_PATH = DATA_DIR / "links.csv"

TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w342"  # medium size

def load_tmdb_mapping():
    """Returns dict: movieId -> tmdbId"""
    df = pd.read_csv(LINKS_PATH)
    mapping = dict(zip(df["movieId"], df["tmdbId"]))
    return mapping

def get_poster_url(tmdb_id, api_key):
    """Fetch poster URL from TMDB using movie tmdbId."""
    if pd.isna(tmdb_id):
        return None
    
    url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}"
    params = {"api_key": api_key}
    
    try:
        r = requests.get(url, params=params, timeout=5)
        r.raise_for_status()
        data = r.json()
        poster_path = data.get("poster_path")
        if poster_path:
            return TMDB_IMG_BASE + poster_path
    except:
        return None
    
    return None
