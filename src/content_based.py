# src/content_based.py
"""
Content-based recommender utilities.
Provides:
- build_tfidf_matrix(movies_df, max_features) -> tfidf, matrix, prepared_df
- recommend(query, movies_df, tfidf_matrix, top_n=10) -> DataFrame with movieId,title,genres
"""

from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def build_tfidf_matrix(movies_df: pd.DataFrame, max_features: int = 5000) -> Tuple[TfidfVectorizer, any, pd.DataFrame]:
    """
    Build TF-IDF vectorizer + matrix from movies DataFrame.
    Expects movies_df to have columns: movieId, title, genres.
    Returns: (tfidf_vectorizer, tfidf_matrix, prepared_movies_df)
    """
    df = movies_df.copy()
    # ensure movieId is present and numeric
    if "movieId" not in df.columns:
        raise ValueError("movies_df must include 'movieId' column")
    df["movieId"] = df["movieId"].astype(int)
    df["title"] = df["title"].fillna("")
    df["genres"] = df["genres"].fillna("")

    # prepare text
    df["content"] = df["title"] + " " + df["genres"].str.replace("|", " ")
    tfidf = TfidfVectorizer(stop_words="english", max_features=max_features)
    tfidf_matrix = tfidf.fit_transform(df["content"])
    return tfidf, tfidf_matrix, df


def recommend(query: str, movies_df: pd.DataFrame, tfidf_matrix, top_n: int = 10) -> pd.DataFrame:
    """
    Return a DataFrame with columns: movieId, title, genres for top_n movies similar to query.
    Behavior:
      - First take substring title matches (case-insensitive) in priority order.
      - If substring matches < top_n, fill remaining slots using TF-IDF cosine similarity.
    """
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel

    # normalize query
    q = str(query).strip()
    if not q:
        return pd.DataFrame(columns=["movieId", "title", "genres"])

    # 1) substring matches (case-insensitive)
    mask = movies_df["title"].str.contains(q, case=False, na=False)
    results = []
    if mask.any():
        # take these first, up to top_n
        substr_df = movies_df[mask][["movieId", "title", "genres"]].reset_index(drop=True)
        for _, r in substr_df.iterrows():
            results.append((int(r["movieId"]), r["title"], r["genres"]))
            if len(results) >= top_n:
                break

    # if we already have enough, return
    if len(results) >= top_n:
        out = pd.DataFrame(results, columns=["movieId", "title", "genres"])
        return out

    # 2) need more -> compute TF-IDF similarity to fill the rest
    # Build a TF-IDF on the movies content (safe for small dataset)
    vect = TfidfVectorizer(stop_words="english", max_features=min(10000, max(1000, tfidf_matrix.shape[1])))
    try:
        # fit on movie contents
        vect_matrix = vect.fit_transform(movies_df["content"])
        q_vec = vect.transform([q])
        sim = linear_kernel(q_vec, vect_matrix).flatten()
    except Exception:
        # As a fallback, return whatever we have so far (no TF-IDF available)
        if results:
            return pd.DataFrame(results, columns=["movieId", "title", "genres"])
        return pd.DataFrame(columns=["movieId", "title", "genres"])

    # rank indices by similarity
    ranked_idx = np.argsort(sim)[::-1]

    # add movies from TF-IDF ranking that are not already in results
    existing_ids = set([r[0] for r in results])
    for idx in ranked_idx:
        mid = int(movies_df.iloc[idx]["movieId"])
        if mid in existing_ids:
            continue
        title = movies_df.iloc[idx]["title"]
        genres = movies_df.iloc[idx]["genres"]
        results.append((mid, title, genres))
        existing_ids.add(mid)
        if len(results) >= top_n:
            break

    out = pd.DataFrame(results, columns=["movieId", "title", "genres"])
    return out


    # transform the query into the tfidf space (use the vectorizer's transform via building a small vector)
    # simpler: compute cosine similarity between query treated as text and the matrix
    # build a temporary tfidf to vectorize the query using same vectorizer vocabulary
    # But we don't have the vectorizer object here; caller may keep it. To keep function simple, we will
    # compute cosine similarity by searching titles for best match and then use neighbors.
    # Safer approach: find the movie titles that best match query (case-insensitive substring),
    # if none found, compute using TF-IDF vectorizer by reconstructing vectorizer from matrix/df is complex.
    # We'll do a hybrid: try exact substring matches first, else use TF-IDF by fitting a small local vectorizer.

    # try substring match (case-insensitive)
    mask = movies_df["title"].str.contains(q, case=False, na=False)
    if mask.any():
        # return the top_n movies where the query appears in title (prioritize shorter distance)
        cand = movies_df[mask].head(top_n)[["movieId", "title", "genres"]]
        return cand.reset_index(drop=True)

    # fallback: compute cosine similarity using TF-IDF
    # We assume tfidf_matrix corresponds to movies_df order
    # Create a small TF-IDF fitted on the same vocabulary by reusing vectorizer from build_tfidf_matrix
    # but here we don't have vectorizer; instead compute similarity by building a vectorizer with same tokens from content
    vect = TfidfVectorizer(stop_words="english", max_features=min(10000, tfidf_matrix.shape[1]))
    # fit on movie contents then transform
    vect_matrix = vect.fit_transform(movies_df["content"])
    q_vec = vect.transform([q])
    sim = linear_kernel(q_vec, vect_matrix).flatten()
    top_idx = np.argsort(sim)[::-1][:top_n]
    recs = movies_df.iloc[top_idx][["movieId", "title", "genres"]]
    return recs.reset_index(drop=True)
