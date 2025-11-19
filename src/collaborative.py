# src/collaborative.py
"""
Sklearn-based collaborative recommender (TruncatedSVD)
Provides run_cf_sklearn(n_factors, user_id, top_n, save_csv=False)
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from utils import load_ratings, load_movies


def run_cf_sklearn(n_factors=50, user_id=1, top_n=10, save_csv=False, csv_path=None):
    print("Loading ratings...")
    ratings = load_ratings()[['userId', 'movieId', 'rating']]
    movies = load_movies()[['movieId', 'title']]

    print("Building user-item matrix...")
    pivot = ratings.pivot_table(index='userId', columns='movieId', values='rating', fill_value=0)

    user_ids = pivot.index.tolist()
    movie_ids = pivot.columns.tolist()
    user_to_index = {u: i for i, u in enumerate(user_ids)}
    movie_to_index = {m: i for i, m in enumerate(movie_ids)}

    n_components = min(n_factors, max(1, len(movie_ids) - 1))
    print(f"Fitting TruncatedSVD (n_components={n_components}) ...")
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    U = svd.fit_transform(pivot)           # (n_users, n_factors)
    Vt = svd.components_                   # (n_factors, n_movies)

    user_factors = U
    item_factors = Vt.T

    def predict(user_idx, movie_idx):
        return float(np.dot(user_factors[user_idx], item_factors[movie_idx]))

    if user_id not in user_to_index:
        print(f"User {user_id} not found in dataset. Available users: {user_ids[:10]} ...")
        return []

    uidx = user_to_index[user_id]
    rated = set(ratings[ratings.userId == user_id].movieId.tolist())
    candidates = [m for m in movie_ids if m not in rated]

    print("Scoring candidates...")
    preds = []
    for mid in candidates:
        midx = movie_to_index[mid]
        score = predict(uidx, midx)
        preds.append((mid, score))

    preds_sorted = sorted(preds, key=lambda x: x[1], reverse=True)[:top_n]
    rows = []
    for mid, score in preds_sorted:
        title = movies.loc[movies.movieId == mid, 'title'].values
        title = title[0] if len(title) else str(mid)
        rows.append({'movieId': int(mid), 'title': title, 'score': score})
    # print
    print(f"\nTop {top_n} recommendations for user {user_id}:")
    for r in rows:
        print(f"{r['title']} (predicted score: {r['score']:.3f})")

    # optionally save CSV
    if save_csv:
        out = pd.DataFrame(rows)
        path = csv_path or f"recs_user{user_id}.csv"
        out.to_csv(path, index=False)
        print(f"\nSaved recommendations to {path}")

    return rows


def parse_args_and_run():
    parser = argparse.ArgumentParser(description="Sklearn SVD collaborative recommender")
    parser.add_argument("--user", type=int, default=1, help="user id to recommend for")
    parser.add_argument("--top", type=int, default=10, help="top-N recommendations")
    parser.add_argument("--factors", type=int, default=50, help="number of latent factors")
    parser.add_argument("--save", action="store_true", help="save results to CSV")
    parser.add_argument("--csv", type=str, default=None, help="output CSV path (optional)")
    args = parser.parse_args()
    run_cf_sklearn(n_factors=args.factors, user_id=args.user, top_n=args.top, save_csv=args.save, csv_path=args.csv)


if __name__ == "__main__":
    parse_args_and_run()
