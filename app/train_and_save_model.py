"""
One-time build script: runs the full pipeline and saves a single artifact
file the Streamlit app can load instantly (no retraining at runtime).

Usage:
    python train_and_save_model.py
"""
import joblib
import config
from data_pipeline import build_dataset
from content_model import build_content_model
from cf_model import build_cf_model


def main():
    print("Loading and cleaning data...")
    df, books = build_dataset()
    print(f"  Ratings: {len(df):,}  Users: {df['User-ID'].nunique():,}  "
          f"Books: {df['ISBN'].nunique():,}")

    print("Building content-based model (TF-IDF)...")
    book_meta, tfidf, tfidf_matrix, book_indices = build_content_model(df, books)

    print("Building collaborative filtering model (SVD)...")
    user2idx, isbn2idx, user_means, U_sigma, Vt = build_cf_model(df)

    rated_by_user = df.groupby("User-ID")["ISBN"].apply(set).to_dict()
    sample_user_ids = df["User-ID"].value_counts().head(config.N_SAMPLE_USERS).index.tolist()

    artifacts = {
        "book_meta": book_meta,
        "tfidf": tfidf,
        "tfidf_matrix": tfidf_matrix,
        "book_indices": book_indices,
        "user2idx": user2idx,
        "isbn2idx": isbn2idx,
        "user_means": user_means,
        "U_sigma": U_sigma,
        "Vt": Vt,
        "rated_by_user": rated_by_user,
        "sample_user_ids": sample_user_ids,
    }

    print(f"Saving artifacts to {config.MODEL_ARTIFACT_PATH} ...")
    joblib.dump(artifacts, config.MODEL_ARTIFACT_PATH, compress=3)
    print("Done.")


if __name__ == "__main__":
    main()