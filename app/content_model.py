"""Content-based recommendations: TF-IDF over title + author + publisher."""
import pandas as pd
import numpy as np
from difflib import get_close_matches
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import config


def build_content_model(df: pd.DataFrame, books: pd.DataFrame):
    """
    Returns:
        book_meta: one row per book that has ratings, with a 'content' column
        tfidf: fitted TfidfVectorizer
        tfidf_matrix: sparse (n_books, n_features) matrix
        book_indices: lowercase title -> row index lookup
    """
    book_meta = books[books["ISBN"].isin(df["ISBN"].unique())].copy()
    book_meta = book_meta.drop_duplicates(subset="ISBN").reset_index(drop=True)

    book_meta["content"] = (
        book_meta["Book-Title"].fillna("") + " " +
        book_meta["Book-Author"].fillna("") + " " +
        book_meta["Publisher"].fillna("")
    ).str.lower()

    tfidf = TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES,
        stop_words="english",
        ngram_range=config.TFIDF_NGRAM_RANGE,
    )
    tfidf_matrix = tfidf.fit_transform(book_meta["content"])

    book_indices = pd.Series(book_meta.index, index=book_meta["Book-Title"].str.lower())
    return book_meta, tfidf, tfidf_matrix, book_indices


def find_seed_index(title: str, book_indices: pd.Series):
    """Look up a book's row index by title, falling back to fuzzy matching."""
    title_lower = title.lower()
    if title_lower in book_indices:
        idx = book_indices[title_lower]
        return idx.iloc[0] if isinstance(idx, pd.Series) else idx

    matches = get_close_matches(title_lower, book_indices.index, n=1, cutoff=0.5)
    if matches:
        idx = book_indices[matches[0]]
        return idx.iloc[0] if isinstance(idx, pd.Series) else idx
    return None


def content_based_recommend(title: str, book_meta, tfidf_matrix, book_indices, n=10):
    """Top-N books most similar to `title` by TF-IDF cosine similarity."""
    idx = find_seed_index(title, book_indices)
    if idx is None:
        return None

    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_indices = np.argsort(sim_scores)[::-1][1:n + 1]

    results = book_meta.iloc[sim_indices][
        ["Book-Title", "Book-Author", "Publisher", "Image-URL-M"]
    ].copy()
    results["similarity_score"] = sim_scores[sim_indices].round(4)
    return results.reset_index(drop=True)