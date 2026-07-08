"""Loading and cleaning the Book-Crossing dataset."""
import pandas as pd
import config


def load_raw_data():
    """Read books.csv and ratings.csv with the correct encoding/delimiter."""
    books = pd.read_csv(
        config.BOOKS_CSV, sep=";", encoding="latin-1",
        on_bad_lines="skip", dtype={"Year-Of-Publication": str},
    )
    ratings = pd.read_csv(
        config.RATINGS_CSV, sep=";", encoding="latin-1", on_bad_lines="skip"
    )
    return books, ratings


def clean_books_metadata(books: pd.DataFrame) -> pd.DataFrame:
    """Fill missing text fields so downstream text-joins never hit NaN."""
    books = books.copy()
    books["Book-Title"] = books["Book-Title"].fillna("Unknown Title").str.strip()
    books["Book-Author"] = books["Book-Author"].fillna("Unknown Author").str.strip()
    books["Publisher"] = books["Publisher"].fillna("Unknown Publisher").str.strip()
    return books


def filter_cold_start(ratings: pd.DataFrame) -> pd.DataFrame:
    """Drop unrated (0) rows, then keep only active users and popular books."""
    explicit = ratings[ratings["Book-Rating"] > 0].copy()

    user_counts = explicit["User-ID"].value_counts()
    active_users = user_counts[user_counts >= config.MIN_RATINGS_PER_USER].index
    explicit = explicit[explicit["User-ID"].isin(active_users)]

    book_counts = explicit["ISBN"].value_counts()
    popular_books = book_counts[book_counts >= config.MIN_RATINGS_PER_BOOK].index
    explicit = explicit[explicit["ISBN"].isin(popular_books)]

    return explicit


def build_dataset():
    """Full pipeline: load -> clean -> filter -> merge. Returns (df, books)."""
    books, ratings = load_raw_data()
    books = clean_books_metadata(books)
    explicit = filter_cold_start(ratings)

    df = explicit.merge(
        books[["ISBN", "Book-Title", "Book-Author", "Publisher", "Image-URL-M"]],
        on="ISBN", how="inner",
    )
    return df, books