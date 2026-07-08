"""Central configuration — tweak model/data parameters here only."""

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data files (expected in the parent folder, one level up from app/)
BOOKS_CSV = "../books.csv"
RATINGS_CSV = "../ratings.csv"

# Cold-start filtering thresholds
MIN_RATINGS_PER_USER = 5
MIN_RATINGS_PER_BOOK = 5

# Content model (TF-IDF)
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)

# Collaborative filtering (SVD)
N_FACTORS = 50
RANDOM_STATE = 42

# Output artifact
MODEL_ARTIFACT_PATH = os.path.join(BASE_DIR, "model_artifacts.pkl")

# How many sample users to expose in the UI dropdown
N_SAMPLE_USERS = 200