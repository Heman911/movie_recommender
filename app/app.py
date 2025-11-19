# app/app.py - patched (safe ordering + debug)
from pathlib import Path
import sys
import os

# standard imports
import pandas as pd

# Resolve project root and src folder reliably
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]   # one above app/ -> project root
SRC_DIR = PROJECT_ROOT / "src"

# Ensure src is on sys.path (use resolved absolute path)
SRC_PATH = str(SRC_DIR.resolve())
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

# Build debug info (safe — uses Path.exists, doesn't import streamlit)
_debug_info = {
    "project_root": str(PROJECT_ROOT),
    "src_path": SRC_PATH,
    "src_exists": SRC_DIR.exists(),
    "content_based_exists": (SRC_DIR / "content_based.py").exists(),
    "collaborative_exists": (SRC_DIR / "collaborative.py").exists(),
    "utils_exists": (SRC_DIR / "utils.py").exists(),
    "sys_path_head": sys.path[0:5],
}

# Safe local imports (these are modules inside src/)
try:
    from content_based import build_tfidf_matrix, recommend as cb_recommend
    from collaborative import run_cf_sklearn
    from utils import load_movies, load_ratings, get_poster_for_movie, check_tmdb_key
except Exception as e:
    # Import failure — show a clear error when running with Streamlit
    try:
        import streamlit as st
        st.set_page_config(page_title="Movie Recommender - Error")
        st.title("Module import error")
        st.error("Failed to import local modules from src/. See debug below.")
        st.exception(e)
        st.write("Debug info:")
        for k, v in _debug_info.items():
            st.write(f"- **{k}**: {v}")
        st.stop()
    except Exception:
        # Not running Streamlit (e.g., running unit tests) — re-raise
        raise

# Now safe to import Streamlit (after imports above)
import streamlit as st

# Poster cache (optional)
POSTER_CACHE = {}
POSTER_CACHE_PATH = PROJECT_ROOT / "data" / "posters_cache.csv"
if POSTER_CACHE_PATH.exists():
    try:
        df_cache = pd.read_csv(POSTER_CACHE_PATH)
        POSTER_CACHE = {int(r["movieId"]): (r["poster"] if pd.notna(r["poster"]) else None)
                        for _, r in df_cache.iterrows()}
    except Exception:
        POSTER_CACHE = {}

# Minimal debug UI (safe)
st.set_page_config(page_title="Movie Recommender", layout="centered")
st.title("🎬 Movie Recommender — Content & Collaborative")

with st.sidebar.expander("Debug info (click)"):
    for k, v in _debug_info.items():
        st.write(f"**{k}**: {v}")

    # Safe TMDB key check (function provided by utils; won't crash)
    try:
        key_present = check_tmdb_key()
        st.write("TMDB key present?", bool(key_present))
    except Exception as e:
        st.write("TMDB key check error:", e)

# Sidebar
st.sidebar.header("Settings")
top_n = st.sidebar.slider("Top N results", 5, 30, 10)
cb_max_features = st.sidebar.slider("TF-IDF max features", 1000, 10000, 5000, step=500)
cf_factors = st.sidebar.slider("SVD factors", 10, 200, 50, step=10)
st.sidebar.write("Put MovieLens ml-latest-small under data/ml-latest-small")

# Poster cache usage inside app functions below
tab1, tab2 = st.tabs(["Content-based", "Collaborative"])

with tab1:
    st.header("Content-based recommender")
    try:
        movies = load_movies()
    except Exception as e:
        st.error(f"Could not load movies.csv: {e}")
        st.stop()

    @st.cache_data(show_spinner=False)
    def build_matrix(movies_df, max_features):
        df = movies_df.copy()
        df['content'] = df['title'].fillna('') + ' ' + df['genres'].fillna('').str.replace('|', ' ')
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf = TfidfVectorizer(stop_words='english', max_features=max_features)
        tfidf_matrix = tfidf.fit_transform(df['content'])
        return tfidf, tfidf_matrix, df

    tfidf, tfidf_matrix, movies_prepared = build_matrix(movies, cb_max_features)

    q = st.text_input("Enter a movie title (exact or partial)", "Toy Story")
    if st.button("Recommend similar movies"):
        if not q.strip():
            st.warning("Enter a movie title.")
        else:
            recs = cb_recommend(q, movies_prepared, tfidf_matrix, top_n=top_n)
            if isinstance(recs, pd.DataFrame) and not recs.empty:
                st.write(f"Top {len(recs)} recommendations for **{q}**:")
                for _, row in recs.reset_index(drop=True).iterrows():
                    movie_id = int(row.get("movieId")) if "movieId" in row else None
                    title = row.get("title", "")
                    genres = row.get("genres", "")

                    poster = None
                    if movie_id is not None:
                        poster = POSTER_CACHE.get(movie_id)
                    if not poster:
                        poster = get_poster_for_movie(movie_id, title)

                    cols = st.columns([1, 4])
                    with cols[0]:
                        if poster:
                            st.image(poster, width=120)
                        else:
                            st.write("No image")
                    with cols[1]:
                        st.markdown(f"**{title}**")
                        st.write(genres)
                        st.write("")
                csv = recs.to_csv(index=False).encode("utf-8")
                st.download_button("Download CSV", csv, f"cb_recs_{q.replace(' ','_')}.csv", "text/csv")
            else:
                st.info("No recommendations found. Try a different or partial title (e.g., 'Godfather').")

with tab2:
    st.header("Collaborative recommender (SVD)")
    try:
        ratings = load_ratings()
    except Exception as e:
        st.error(f"Could not load ratings.csv: {e}")
        st.stop()

    st.write(f"Dataset contains {ratings.userId.nunique()} users and {ratings.movieId.nunique()} movies.")
    user_id = st.number_input("User ID", min_value=int(ratings.userId.min()), max_value=int(ratings.userId.max()), value=1, step=1)
    if st.button("Recommend for user"):
        with st.spinner("Computing recommendations..."):
            rows = run_cf_sklearn(n_factors=cf_factors, user_id=int(user_id), top_n=top_n, save_csv=False)
            if rows:
                df = pd.DataFrame(rows)
                # enrich with posters
                posters = []
                for r in df.to_dict(orient="records"):
                    mid = int(r.get("movieId"))
                    title = r.get("title", "")
                    poster = POSTER_CACHE.get(mid)
                    if not poster:
                        poster = get_poster_for_movie(mid, title)
                    posters.append(poster)
                df["poster"] = posters

                for _, row in df.iterrows():
                    cols = st.columns([1, 4])
                    with cols[0]:
                        if pd.notna(row["poster"]):
                            st.image(row["poster"], width=120)
                        else:
                            st.write("No image")
                    with cols[1]:
                        st.markdown(f"**{row['title']}**")
                        st.write(f"Predicted score: {row['score']:.3f}")
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download CSV", csv, f"cf_recs_user{user_id}.csv", "text/csv")
            else:
                st.info("No recommendations (user may not exist).")

st.markdown("---")
st.caption("Built with MovieLens ml-latest-small. Put the unzipped folder at data/ml-latest-small.")
