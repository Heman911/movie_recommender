# scripts/fetch_posters.py
import time
import pandas as pd
from src.utils import load_movies, get_poster_for_movie

movies = load_movies()
rows = []
for _, r in movies.iterrows():
    mid = int(r.movieId)
    title = r.title
    poster = get_poster_for_movie(mid, title)
    rows.append({"movieId": mid, "title": title, "poster": poster})
    # be polite to TMDB
    time.sleep(0.25)

pd.DataFrame(rows).to_csv("data/posters_cache.csv", index=False)
print("Wrote data/posters_cache.csv")
