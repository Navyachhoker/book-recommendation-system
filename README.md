# Book Recommendation System

A hybrid recommendation engine built on the Book-Crossing dataset (1.15M ratings, 271K books, 278K users), combining content-based filtering (TF-IDF + cosine similarity) with collaborative filtering (bias-aware SVD). Deployed as an interactive Streamlit web application.

**Live demo:** https://book-recommendation-system-s37uayimdoxbuvdgxnlmto.streamlit.app/
**Repository:** https://github.com/Navyachhoker/book-recommendation-system

---

## Results

The collaborative filtering model went through two iterations. The first (a mean-centred Truncated SVD) modeled only *user* rating bias. Evaluating it against naive baselines revealed it didn't clearly outperform a simple per-user-average — a common failure mode in sparse-data CF caused by missing *item* bias (some books are just rated more highly than others, regardless of who's reading them). Switching to a bias-aware SVD (via the `surprise` library, modeling user bias, item bias, and latent factors jointly with L2 regularization) closed that gap and materially improved every metric.

| Model | RMSE | MAE | Precision@5 | Recall@5 |
|---|---|---|---|---|
| Global mean baseline | 1.781 | 1.419 | — | — |
| Per-user mean baseline | 1.652 | 1.246 | — | — |
| Mean-centred SVD (v1) | 1.841 ± 0.011 | 1.322 ± 0.009 | 0.614 | 0.814 |
| **Bias-aware SVD (v2, current)** | **1.570 ± 0.008** | **1.211 ± 0.005** | **0.751** | 0.751 |

*RMSE/MAE reported as 5-fold CV mean ± std, evaluated on a held-out 20% test split (27,443 pairs). Precision/Recall use a 7/10 "liked" threshold.*

**Why this matters:** the first model looked reasonable in isolation (RMSE 1.84 "sounds fine" on a 1–10 scale) but didn't actually beat a one-line baseline. Rather than stopping there, the model was diagnosed, re-architected to include item bias, and re-validated — a ~15% RMSE improvement over v1, and now a clear, defensible win over both baselines.

---

## Overview

Recommendation systems face two structural problems: the **cold-start problem** (no history for new users or books) and **data sparsity** (most users rate a tiny fraction of the catalog). This project addresses both directly:

- **Content-based filtering** solves cold-start by recommending on book metadata (title, author, publisher) alone — no rating history required.
- **Collaborative filtering** captures patterns invisible to content matching (readers who like X also like Y, even when X and Y share no metadata) by factorizing a 12,873 × 10,811 user-item matrix, which is 99.997% sparse, into 50 latent factors.
- A **hybrid layer** blends both signals with a tunable weight, so recommendations stay useful whether or not a user has rating history.

---

## Methodology

### Data pipeline

- Source: Book-Crossing dataset (271,360 books, 1,149,780 ratings, 278,858 users)
- Removed 716,109 implicit (unrated, score = 0) interactions, keeping 433,671 explicit ratings
- Applied cold-start filtering (users and books with fewer than 5 ratings dropped), reducing to 141,081 ratings across 13,030 users and 11,234 books
- Cleaned and merged metadata (title, author, publisher, year), producing a final working set of 137,214 rating-metadata pairs

### Content-based model

- Combined title, author, and publisher into a single text field per book
- Vectorized with TF-IDF (5,000 features, unigrams + bigrams, English stop-words removed) over 10,811 unique books
- Computed pairwise cosine similarity to rank similar titles for any given book

### Collaborative filtering model

Two implementations, kept side by side in the codebase (`cf_model.py`):

**v1 — Mean-centred Truncated SVD** *(legacy, still used as an automatic fallback)*
- Built a sparse user-item rating matrix (12,873 users × 10,811 books)
- Mean-centred per user to remove individual rating-scale bias
- Applied randomized Truncated SVD with 50 latent factors
- Predicted ratings via the dot product of user and book latent vectors, clipped to [1, 10]

**v2 — Bias-aware SVD** *(current default, via `surprise`)*
- Jointly models global mean, per-user bias, per-item bias, and 50 latent factors, trained with stochastic gradient descent and L2 regularization (`reg_all=0.05`, 20 epochs)
- Explicitly captures item bias — the missing piece in v1 — which is why it closes the gap against the per-user-mean baseline
- The app loads whichever model is available in the saved artifacts; if the bias-aware model is present, it's used automatically, with v1 as a transparent fallback if not

### Hybrid layer

- Combines normalized CF and content scores using a weighted average (default alpha = 0.6 favoring CF), user-adjustable in the app
- Falls back to content-based scoring when a user has no rating history

---

## How It Works

Example: recommending for an existing user based on their rating history and a seed book they liked.

```
Input:  user_id = 11676

Collaborative filtering output (top 5, by predicted rating):
  Angels & Demons                        — Dan Brown            (7.59)
  A Case of Need                         — Michael Crichton     (7.59)
  The Girlfriends' Guide to Pregnancy    — Vicki Iovine          (7.58)
  Don't Stand Too Close to a Naked Man   — Tim Allen             (7.58)
  The Poisonwood Bible: A Novel          — Barbara Kingsolver    (7.58)
```

The hybrid layer re-ranks these using a weighted blend of the CF score and a content-similarity score against the user's seed book, surfacing titles that are strong on both dimensions rather than just one.

---

## Features

**Content-Based Recommendations**
Recommends books similar to a selected title using TF-IDF + cosine similarity over title, author, and publisher.

**Collaborative Filtering**
Generates personalized recommendations from user-book rating interactions via bias-aware SVD.

**Hybrid Recommendation Engine**
Combines both signals with a user-adjustable weight for more relevant, cold-start-resilient results.

**Interactive Streamlit UI**
Search books by title, select existing users, adjust recommendation count and hybrid weighting, view book covers and metadata.

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn (TF-IDF Vectorizer, Cosine Similarity, Truncated SVD), Surprise (bias-aware SVD) |
| Deployment | Streamlit (Streamlit Community Cloud) |

---

## Dataset

Built on the [Book-Crossing Dataset](http://www2.informatik.uni-freiburg.de/~cziegler/BX/):

| File | Description |
|---|---|
| `Books.csv` | Book metadata — ISBN, title, author, publisher, cover images |
| `Ratings.csv` | User ratings for books (explicit, 1–10, and implicit, 0) |
| `Users.csv` | User information and demographics |

---

## Project Structure

```
book-recommendation-system/
├── app.py                     # Streamlit application entry point
├── config.py                  # Configuration and constants
├── data_pipeline.py           # Data loading, cleaning, cold-start filtering
├── content_model.py           # TF-IDF + cosine similarity logic
├── cf_model.py                # Mean-centred SVD (v1) + bias-aware SVD (v2, surprise)
├── hybrid_model.py            # Weighted combination of both models
├── train_and_save_model.py    # Offline training pipeline, produces model_artifacts.pkl
├── model_artifacts.pkl        # Serialized trained models
├── notebooks/
│   └── model_evaluation.ipynb # Baseline comparison, CV, precision/recall analysis
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/Navyachhoker/book-recommendation-system.git
cd book-recommendation-system
pip install -r requirements.txt
python train_and_save_model.py   # builds model_artifacts.pkl (requires Books.csv / Ratings.csv)
streamlit run app.py
```

---

## Model Evaluation Notes

- Evaluation used a held-out 80/20 train-test split, validated with 5-fold cross-validation to confirm stability.
- Ratings ≥ 7/10 were treated as the "liked" threshold for ranking metrics.
- Baselines (global mean, per-user mean) were computed on the same split to contextualize whether the learned model outperforms trivial heuristics — a check that revealed a real limitation in v1 and directly motivated the move to a bias-aware model in v2.
- The precision/recall shift from v1 to v2 (higher precision, slightly lower recall) reflects the new model making more conservative, higher-confidence recommendations rather than a straightforward "better in every way" result — a normal precision/recall tradeoff, not a modeling error.

---

## Future Enhancements

- Replace TF-IDF with transformer-based embeddings (e.g., sentence-transformers) for semantic content matching
- Incorporate book descriptions and user reviews as additional content signal
- Add implicit-feedback modeling (currently 716K implicit interactions are dropped entirely)
- Hyperparameter tuning (grid search over `n_factors`, `reg_all`) to push the bias-aware SVD further
- LLM-generated natural-language explanations for why each book was recommended

---

## Author

**Navya Chhoker**
GitHub: [github.com/Navyachhoker](https://github.com/Navyachhoker)
