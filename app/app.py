"""Streamlit UI — loads precomputed artifacts and exposes 3 recommendation modes."""
import joblib
import streamlit as st
import config
from content_model import content_based_recommend
from cf_model import cf_recommend
from hybrid_model import hybrid_recommend

st.set_page_config(page_title="Book Recommender", page_icon="📚", layout="wide")


@st.cache_resource
def load_artifacts():
    return joblib.load(config.MODEL_ARTIFACT_PATH)


A = load_artifacts()


def render_results(df, score_col, score_label):
    if df is None or df.empty:
        st.warning("No matches found. Try a different title or user.")
        return
    cols = st.columns(5)
    for i, row in df.iterrows():
        with cols[i % 5]:
            if row.get("Image-URL-M"):
                st.image(row["Image-URL-M"], use_container_width=True)
            st.markdown(f"**{row['Book-Title']}**")
            st.caption(row["Book-Author"])
            st.caption(f"{score_label}: {row[score_col]}")


st.title("📚 Book Recommendation System")
st.caption("Content-based (TF-IDF + cosine similarity) and collaborative filtering "
           "(SVD) on the Book-Crossing dataset.")

tab1, tab2, tab3 = st.tabs(["🔎 Content-Based", "👤 Collaborative Filtering", "🧬 Hybrid"])

with tab1:
    st.subheader("Find books similar to one you liked")
    title = st.text_input("Book title", value="The Da Vinci Code", key="cb_title")
    n = st.slider("Number of recommendations", 5, 20, 10, key="cb_n")
    if st.button("Get recommendations", key="cb_btn"):
        results = content_based_recommend(title, A["book_meta"], A["tfidf_matrix"],
                                           A["book_indices"], n)
        render_results(results, "similarity_score", "Similarity")

with tab2:
    st.subheader("Personalized picks for an existing user")
    st.caption("Sample user IDs from the real Book-Crossing dataset.")
    user_id = st.selectbox("User ID", A["sample_user_ids"], key="cf_user")
    n2 = st.slider("Number of recommendations", 5, 20, 10, key="cf_n")
    if st.button("Get recommendations", key="cf_btn"):
        results = cf_recommend(user_id, A["book_meta"], A["isbn2idx"], A["rated_by_user"],
                                A["user2idx"], A["user_means"], A["U_sigma"], A["Vt"], n2)
        render_results(results, "predicted_rating", "Predicted rating")

with tab3:
    st.subheader("Combine a user's taste profile with a seed book")
    user_id3 = st.selectbox("User ID", A["sample_user_ids"], key="hy_user")
    seed = st.text_input("Seed book (a title they liked)", value="Classical Mythology", key="hy_seed")
    alpha = st.slider("Weight on collaborative filtering (vs. content)", 0.0, 1.0, 0.6, key="hy_alpha")
    n3 = st.slider("Number of recommendations", 5, 20, 10, key="hy_n")
    if st.button("Get recommendations", key="hy_btn"):
        results = hybrid_recommend(user_id3, seed, A["book_meta"], A["tfidf_matrix"],
                                    A["book_indices"], A["isbn2idx"], A["rated_by_user"],
                                    A["user2idx"], A["user_means"], A["U_sigma"], A["Vt"],
                                    n3, alpha)
        render_results(results, "hybrid_score", "Hybrid score")

st.divider()
st.caption("Built with TF-IDF cosine similarity (content) and truncated SVD (collaborative filtering) "
           "on the Book-Crossing dataset.")