# src/utils.py  (patched for robust secrets + poster cache)
import os
import csv
from pathlib import Path
from typing import Optional, Dict

import pandas as pd
import requests

# Try to import streamlit, but do NOT access st.secrets at import-time without guard.
try:
    import streamlit as st  # type: ignore
    _HAS_STREAMLIT = True
except Exception:
    st = None
    _HAS_STREAMLIT = False

# project root: one above src/
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "ml-latest-small"
MOVIES_CSV = DATA_DIR / "movies.csv"
RATINGS_CSV = DATA_DIR / "ratings.csv"
LINKS_CSV = DATA_DIR / "links.csv"

TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w500"
TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_MOVIE_URL = "https://api.themoviedb.org/3/movie/{}"

# Poster cache file (persisted)
POSTER_CACHE_CSV = ROOT / "data" / "posters_cache.csv"
POSTER_CACHE: Dict[int, Optional[str]] = {}

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
# Poster cache helpers
# ---------------------------
def _load_poster_cache():
    global POSTER_CACHE
    POSTER_CACHE = {}
    try:
        if POSTER_CACHE_CSV.exists():
            df = pd.read_csv(POSTER_CACHE_CSV)
            for _, r in df.iterrows():
                try:
                    POSTER_CACHE[int(r["movieId"])] = r["poster"] if pd.notna(r["poster"]) else None
                except Exception:
                    continue
    except Exception:
        POSTER_CACHE = {}


def _save_poster_cache():
    try:
        POSTER_CACHE_CSV.parent.mkdir(parents=True, exist_ok=True)
        with open(POSTER_CACHE_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["movieId", "poster"])
            writer.writeheader()
            for mid, poster in POSTER_CACHE.items():
                writer.writerow({"movieId": mid, "poster": poster})
    except Exception:
        pass


# initialize in-memory cache at import time
_load_poster_cache()

# ---------------------------
# TMDB helpers
# ---------------------------
def _get_tmdb_key() -> Optional[str]:
    """
    Safe retrieval:
      - Try Streamlit secrets (if available) inside guarded block
      - If not found, attempt environment var (and .env if present)
    """
    key = None
    if _HAS_STREAMLIT:
        try:
            # Accessing st.secrets can raise StreamlitSecretNotFoundError locally;
            # wrap in try/except and don't re-raise.
            key = st.secrets.get("TMDB_API_KEY")
        except Exception:
            key = None

    if not key:
        # fallback to environment or .env (local dev)
        key = os.getenv("TMDB_API_KEY")
        if not key:
            # attempt to load local .env if present (no harm on cloud)
            try:
                from dotenv import load_dotenv
                load_dotenv(ROOT / ".env")
                key = os.getenv("TMDB_API_KEY")
            except Exception:
                key = None
    return key


def check_tmdb_key() -> bool:
    """Utility for app to check whether a key is available (safe)."""
    return bool(_get_tmdb_key())


def load_tmdb_map() -> dict:
    """Return mapping movieId -> tmdbId (ints), cached by re-reading links.csv each call."""
    links = load_links()
    mapping = {}
    for _, r in links.iterrows():
        mid = int(r["movieId"])
        try:
            tmdb = int(r["tmdbId"])
        except Exception:
            tmdb = None
        mapping[mid] = tmdb
    return mapping


def _get_poster_from_tmdb_id(tmdb_id: int) -> Optional[str]:
    key = _get_tmdb_key()
    if not key:
        return None
    try:
        resp = requests.get(TMDB_MOVIE_URL.format(int(tmdb_id)), params={"api_key": key}, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        poster_path = data.get("poster_path")
        if poster_path:
            return TMDB_IMG_BASE + poster_path
        return None
    except Exception:
        return None


def _search_movie_poster_by_title(title: str) -> Optional[str]:
    key = _get_tmdb_key()
    if not key:
        return None
    try:
        resp = requests.get(TMDB_SEARCH_URL, params={"api_key": key, "query": title, "include_adult": "false"}, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results") or []
        if not results:
            return None
        poster_path = results[0].get("poster_path")
        if poster_path:
            return TMDB_IMG_BASE + poster_path
        return None
    except Exception:
        return None


def get_poster_for_movie(movieId: int, title: str = None, persist: bool = True) -> Optional[str]:
    """
    Robust fetcher:
      - return cached URL if present
      - try TMDB by tmdbId (links.csv)
      - fallback to search by title
      - update cache (in-memory and optionally persisted)
    """
    # ensure cache loaded
    if not POSTER_CACHE:
        _load_poster_cache()

    try:
        mid = int(movieId) if movieId is not None else None
    except Exception:
        mid = None

    # 1) in-memory cache
    if mid is not None and mid in POSTER_CACHE and POSTER_CACHE[mid]:
        return POSTER_CACHE[mid]

    # 2) try tmdb id via links.csv
    try:
        tmdb_map = load_tmdb_map()
    except Exception:
        tmdb_map = {}
    tmdb_id = tmdb_map.get(mid) if mid is not None else None

    if tmdb_id:
        url = _get_poster_from_tmdb_id(tmdb_id)
        if url:
            if mid is not None:
                POSTER_CACHE[mid] = url
                if persist:
                    _save_poster_cache()
            return url

    # 3) fallback: search by title
    if title:
        url = _search_movie_poster_by_title(title)
        if url:
            if mid is not None:
                POSTER_CACHE[mid] = url
                if persist:
                    _save_poster_cache()
            return url

    # 4) nothing found
    if _HAS_STREAMLIT:
        try:
            st.write(f"Poster not found for movieId={movieId}, title='{title}'")
        except Exception:
            pass
    return None
