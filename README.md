# 📚 Book Recommendation System

A content-based book recommendation system built with Python that suggests similar books based on metadata like title, author, and publisher using **TF-IDF Vectorization** and **Cosine Similarity**.

---

## 📌 Project Overview

This project analyzes a large book dataset (Books, Ratings, and Users) and recommends books similar to a given title. It uses Natural Language Processing (NLP) techniques to extract meaningful features from book metadata and compute similarity scores between books.

---

## 📂 Dataset

The project uses the **Book-Crossing Dataset**, which contains three files:

| File | Description |
|------|-------------|
| `books.csv` | Book details — ISBN, Title, Author, Year, Publisher, Image URLs |
| `ratings.csv` | User ratings for books (scale 0–10) |
| `users.csv` | User demographics — ID, Location, Age |

---

## 🛠️ Tech Stack

- **Python 3**
- **Pandas** — Data loading and cleaning
- **NumPy** — Numerical operations
- **Scikit-learn** — TF-IDF Vectorizer, Cosine Similarity

---

## ⚙️ How It Works

1. **Data Loading** — Reads all three CSV files with proper encoding (`latin-1`) and semicolon delimiter
2. **Data Cleaning**
   - Removes duplicate entries
   - Drops rows with missing values
   - Filters invalid publication years (keeps > 1900)
   - Filters unrealistic user ages (keeps 5–90)
   - Removes books with a rating of 0 (unrated)
3. **Feature Engineering** — Combines `Book-Title`, `Book-Author`, and `Publisher` into a single `features` column
4. **TF-IDF Vectorization** — Converts text features into a numerical matrix
5. **Cosine Similarity** — Computes pairwise similarity between all books
6. **Recommendation** — Given a book title, returns top 10 most similar books

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/book-recommendation-system.git
cd book-recommendation-system
```

### 2. Install dependencies

```bash
pip install pandas numpy scikit-learn
```

### 3. Run the notebooks

Open the Jupyter notebook:

- 'book recommendation(content + collaborative filtering).ipynb'

```bash
jupyter notebook
```

### 4. Get recommendations

```python
recommend_books("Bluebeard's Egg and Other Stories")
```

---

## 📁 Project Structure

```
book-recommendation-system/
│
├── books.csv                  # Book metadata
├── ratings.csv                # User ratings
├── users.csv                  # User demographics
├── book recommendation(content + collaborative filtering).ipynb 
└── README.md
```

---

## 📊 Sample Output

```
recommend_books("Bluebeard's Egg and Other Stories")

          Book-Title                        Book-Author
The Handmaid's Tale               Margaret Atwood
Cat's Eye                         Margaret Atwood
...
```

---

## 🔮 Future Improvements

- Add collaborative filtering using user ratings
- Build a web UI with Streamlit or Flask
- Deploy as an API using FastAPI
- Incorporate deep learning embeddings for better recommendations

---
