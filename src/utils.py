# src/utils.py
import os
from pathlib import Path
import pandas as pd
import requests
from dotenv import load_dotenv

# load .env from project root
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=ROOT / ".env")

TMDB_API_KEY = os.getenv("TMDB_API_KEY")  # must put key in .env: TMDB_API_KEY=xxxx

DATA_DIR = ROOT / "data" / "ml-latest-small"
MOVIES_CSV = DATA_DIR / "movies.csv"
RATINGS_CSV = DATA_DIR / "ratings.csv"
LINKS_CSV = DATA_DIR / "links.csv"

TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w500"
TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_MOVIE_URL = "https://api.themoviedb.org/3/movie/{}"

# ---------------------------
# Data loaders
# ---------------------------
def load_movies(path: str = None) -> pd.DataFrame:
    path = Path(path) if path else MOVIES_CSV
    if not path.exists():
        raise FileNotFoundError(f"movies.csv not found at {path}. Download ml-latest-small and put it at data/")
    return pd.read_csv(path)


def load_ratings(path: str = None) -> pd.DataFrame:
    path = Path(path) if path else RATINGS_CSV
    if not path.exists():
        raise FileNotFoundError(f"ratings.csv not found at {path}. Download ml-latest-small and put it at data/")
    return pd.read_csv(path)


def load_links(path: str = None) -> pd.DataFrame:
    path = Path(path) if path else LINKS_CSV
    if not path.exists():
        raise FileNotFoundError(f"links.csv not found at {path}. Download ml-latest-small and put it at data/")
    return pd.read_csv(path)


# ---------------------------
# TMDB poster helpers
# ---------------------------
def load_tmdb_map() -> dict:
    """Return mapping movieId -> tmdbId (ints), cached by re-reading links.csv each call."""
    links = load_links()
    # some tmdbId may be NaN; convert to int when possible
    mapping = {}
    for _, r in links.iterrows():
        mid = int(r["movieId"])
        try:
            tmdb = int(r["tmdbId"])
        except Exception:
            tmdb = None
        mapping[mid] = tmdb
    return mapping


def _get_poster_from_tmdb_id(tmdb_id: int) -> str | None:
    """Fetch poster_path from TMDB movie details using tmdb id; return full image URL or None."""
    if not TMDB_API_KEY:
        # Not fatal — return None and let UI show fallback
        return None
    try:
        resp = requests.get(TMDB_MOVIE_URL.format(int(tmdb_id)), params={"api_key": TMDB_API_KEY}, timeout=6)
        resp.raise_for_status()
        data = resp.json()
        poster_path = data.get("poster_path")
        return TMDB_IMG_BASE + poster_path if poster_path else None
    except Exception:
        return None


def _search_movie_poster_by_title(title: str) -> str | None:
    """Fallback: search TMDB by title and return first poster URL (if any)."""
    if not TMDB_API_KEY:
        return None
    try:
        resp = requests.get(TMDB_SEARCH_URL, params={"api_key": TMDB_API_KEY, "query": title}, timeout=6)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results") or []
        if not results:
            return None
        poster_path = results[0].get("poster_path")
        return TMDB_IMG_BASE + poster_path if poster_path else None
    except Exception:
        return None


def get_poster_for_movie(movieId: int, title: str = None) -> str | None:
    """
    Robust fetcher: try tmdbId first (with retries), then fallback to title search (with retries).
    """
    try:
        tmdb_map = load_tmdb_map()
    except Exception:
        tmdb_map = {}

    tmdb_id = tmdb_map.get(int(movieId)) if movieId is not None else None

    # --- Try tmdbId lookup first ---
    if tmdb_id:
        for _ in range(2):     # retry twice
            url = _get_poster_from_tmdb_id(tmdb_id)
            if url:
                return url

    # --- Fallback: search by movie title ---
    if title:
        for _ in range(2):     # retry twice
            url = _search_movie_poster_by_title(title)
            if url:
                return url

    # Nothing found
    return None
