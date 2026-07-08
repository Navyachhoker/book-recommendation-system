"""Collaborative filtering: mean-centred truncated SVD on the user-item matrix."""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.utils.extmath import randomized_svd
import config


def build_cf_model(df: pd.DataFrame):
    """
    Returns:
        user2idx, isbn2idx: ID -> matrix-row/col lookups
        user_means: per-user average rating (for mean-centring / de-centring)
        U_sigma: (n_users, k) user embeddings weighted by singular values
        Vt: (k, n_books) book embeddings
    """
    all_users = df["User-ID"].unique()
    all_isbns = df["ISBN"].unique()
    user2idx = {u: i for i, u in enumerate(all_users)}
    isbn2idx = {b: i for i, b in enumerate(all_isbns)}

    n_u, n_b = len(all_users), len(all_isbns)
    df = df.copy()
    df["u_idx"] = df["User-ID"].map(user2idx)
    df["b_idx"] = df["ISBN"].map(isbn2idx)

    R = csr_matrix(
        (df["Book-Rating"].values.astype(float), (df["u_idx"].values, df["b_idx"].values)),
        shape=(n_u, n_b),
    )

    user_means = np.zeros(n_u)
    for u in range(n_u):
        row = R.getrow(u)
        if row.nnz > 0:
            user_means[u] = row.data.mean()

    R_centred = R.copy().astype(float)
    for u in range(n_u):
        if R_centred.getrow(u).nnz > 0:
            R_centred.data[R_centred.indptr[u]:R_centred.indptr[u + 1]] -= user_means[u]

    U, sigma, Vt = randomized_svd(
        R_centred, n_components=config.N_FACTORS, random_state=config.RANDOM_STATE
    )
    U_sigma = U * sigma

    return user2idx, isbn2idx, user_means, U_sigma, Vt


def predict_all_ratings(user_id, user2idx, user_means, U_sigma, Vt):
    """Predicted rating for every book, for a given user. None if unknown user."""
    if user_id not in user2idx:
        return None
    u_idx = user2idx[user_id]
    preds = user_means[u_idx] + U_sigma[u_idx] @ Vt
    return np.clip(preds, 1.0, 10.0)


def cf_recommend(user_id, book_meta, isbn2idx, rated_by_user,
                  user2idx, user_means, U_sigma, Vt, n=10):
    """Top-N unread books for a user, ranked by predicted rating."""
    all_preds = predict_all_ratings(user_id, user2idx, user_means, U_sigma, Vt)
    if all_preds is None:
        return None

    rated = rated_by_user.get(user_id, set())
    cand = book_meta.copy()
    cand["b_idx"] = cand["ISBN"].map(isbn2idx)
    cand = cand.dropna(subset=["b_idx"])
    cand["b_idx"] = cand["b_idx"].astype(int)
    cand = cand[~cand["ISBN"].isin(rated)]
    cand["predicted_rating"] = all_preds[cand["b_idx"].values].round(3)

    return (cand[["Book-Title", "Book-Author", "Image-URL-M", "predicted_rating"]]
            .sort_values("predicted_rating", ascending=False)
            .head(n).reset_index(drop=True))