"""Hybrid recommendations: blend CF predicted rating with content similarity."""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from content_model import find_seed_index
from cf_model import predict_all_ratings


def hybrid_recommend(user_id, seed_title, book_meta, tfidf_matrix, book_indices,
                      isbn2idx, rated_by_user, user2idx, user_means, U_sigma, Vt,
                      n=10, alpha=0.6):
    """
    alpha weights collaborative filtering vs. content similarity:
    alpha=1.0 -> pure CF, alpha=0.0 -> pure content-based.
    """
    all_preds = predict_all_ratings(user_id, user2idx, user_means, U_sigma, Vt)
    if all_preds is None:
        return None

    seed_idx = find_seed_index(seed_title, book_indices)
    if seed_idx is not None:
        content_scores = cosine_similarity(tfidf_matrix[seed_idx], tfidf_matrix).flatten()
        isbn_content = dict(zip(book_meta["ISBN"], content_scores))
    else:
        isbn_content = {}

    rated = rated_by_user.get(user_id, set())
    cand = book_meta.copy()
    cand["b_idx"] = cand["ISBN"].map(isbn2idx)
    cand = cand.dropna(subset=["b_idx"])
    cand["b_idx"] = cand["b_idx"].astype(int)
    cand = cand[~cand["ISBN"].isin(rated)]

    cf_vals = all_preds[cand["b_idx"].values]
    cf_min, cf_max = cf_vals.min(), cf_vals.max()
    cf_norm = (cf_vals - cf_min) / (cf_max - cf_min) if cf_max > cf_min else np.zeros_like(cf_vals)

    cand = cand.copy()
    cand["cf_score_norm"] = cf_norm
    cand["content_score"] = cand["ISBN"].map(isbn_content).fillna(0.0)
    cand["hybrid_score"] = alpha * cand["cf_score_norm"] + (1 - alpha) * cand["content_score"]

    return (cand[["Book-Title", "Book-Author", "Image-URL-M", "hybrid_score"]]
            .sort_values("hybrid_score", ascending=False)
            .head(n).reset_index(drop=True))