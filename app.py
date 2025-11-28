# ======================================================
# FINAL APP.PY — MOVIE RECOMMENDER + CSV GRAPH SYSTEM
# 100% WORKING FOR STREAMLIT CLOUD
# ======================================================

import os
import traceback
import streamlit as st
import pandas as pd
import joblib
import requests
from PIL import Image
from io import BytesIO
import altair as alt

st.set_page_config(page_title="Movie Recommender", layout="wide")


# ==========================================================
# ----------------  MODEL LOADING  -------------------------
# ==========================================================

MODEL_DIR = "./recommender_models"
REQUIRED = ["tfidf_vectorizer.joblib", "cosine_sim.joblib",
            "movies_metadata_df.pkl", "title_indices.pkl"]

def check_models(path):
    return os.path.isdir(path) and all(os.path.exists(os.path.join(path, f)) for f in REQUIRED)

def load_models(path):
    try:
        tfidf = joblib.load(os.path.join(path, "tfidf_vectorizer.joblib"))
        cosine_sim = joblib.load(os.path.join(path, "cosine_sim.joblib"))
        md = pd.read_pickle(os.path.join(path, "movies_metadata_df.pkl"))
        indices = pd.read_pickle(os.path.join(path, "title_indices.pkl"))
        return tfidf, cosine_sim, md, indices, None
    except Exception as e:
        return None, None, None, None, traceback.format_exc()


# Load model
ok = check_models(MODEL_DIR)
if ok:
    tfidf, cosine_sim, md, indices, err = load_models(MODEL_DIR)
    if err:
        ok = False

if not ok:
    st.error("❌ Required model files missing. Upload required model files in /recommender_models/")
    st.stop()

if "title" not in md.columns:
    st.error("❌ movies_metadata_df.pkl does not contain 'title' column.")
    st.stop()


# ==========================================================
# -----------------  RECOMMENDER LOGIC  ---------------------
# ==========================================================

def recommend(title, topn=10):
    if title in indices.index:
        idx = indices[title]
    else:
        df = md[md["title"].str.contains(title, case=False, na=False)]
        if df.empty:
            return None
        idx = df.index[0]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:topn+1]

    movie_indexes = [i[0] for i in sim_scores]
    result = md.iloc[movie_indexes][["title", "release_date"]].copy()
    result["score"] = [round(i[1], 4) for i in sim_scores]
    return result.reset_index(drop=True)


# ==========================================================
# ------------------  UI LAYOUT  ---------------------------
# ==========================================================

st.markdown("<h1 style='color:#111;'>🎬 Movie Recommender</h1>", unsafe_allow_html=True)

titles = sorted(md["title"].dropna().unique().tolist())

st.markdown("### 🔍 Search Movie")
col1, col2 = st.columns([3, 1])

with col1:
    selected = st.selectbox("Select a movie:", ["-- choose --"] + titles)
    topn = st.slider("Number of recommendations", 5, 20, 10)

with col2:
    st.info("Uses TF-IDF + Cosine Similarity\nFinds similar movies based on description.")

btn = st.button("Recommend")

if btn:
    if selected == "-- choose --":
        st.warning("Please select a valid movie.")
    else:
        result = recommend(selected, topn)
        if result is None:
            st.error("No recommendations found.")
        else:
            st.success(f"Top {len(result)} recommendations for **{selected}**")
            st.dataframe(result)

            # Bar Chart of similarity score
            try:
                chart = (
                    alt.Chart(result)
                    .mark_bar()
                    .encode(
                        x=alt.X("title:N", sort="-y"),
                        y="score:Q",
                        tooltip=["title", "score"]
                    )
                )
                st.altair_chart(chart, use_container_width=True)
            except:
                st.warning("Chart could not be generated.")


# ==========================================================
# ===============  CSV UPLOAD + GRAPH SYSTEM  ===============
# ==========================================================

st.markdown("---")
st.markdown("## 📁 Upload Your CSV File")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.success("CSV Uploaded Successfully!")
    
    st.markdown("### 🔍 CSV Preview")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if numeric_cols:

        st.markdown("### 📊 Create Graph from CSV")

        selected_col = st.selectbox("Choose numeric column:", numeric_cols)
        graph_type = st.radio("Choose graph type:", ["Bar Chart", "Line Chart", "Area Chart"])

        chart_df = df.reset_index().rename(columns={"index": "Row"})

        # Create charts
        if graph_type == "Bar Chart":
            chart = alt.Chart(chart_df).mark_bar().encode(
                x="Row:O", y=f"{selected_col}:Q", tooltip=[selected_col]
            )
        elif graph_type == "Line Chart":
            chart = alt.Chart(chart_df).mark_line(point=True).encode(
                x="Row:O", y=f"{selected_col}:Q", tooltip=[selected_col]
            )
        else:
            chart = alt.Chart(chart_df).mark_area().encode(
                x="Row:O", y=f"{selected_col}:Q", tooltip=[selected_col]
            )

        st.altair_chart(chart, use_container_width=True)

        # Stats
        st.markdown("### 📌 Statistics")
        st.write(df[selected_col].describe())

    else:
        st.warning("⚠ No numeric columns found in CSV. Cannot generate graph.")


# -------- END --------
