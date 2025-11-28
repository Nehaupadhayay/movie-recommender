# ==========================================================
# FINAL APP.PY — Movie Recommender + CSV Upload + Graphs
# ==========================================================

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

MODEL_DIR = "./recommender_models"
REQUIRED = ["tfidf_vectorizer.joblib", "cosine_sim.joblib",
            "movies_metadata_df.pkl", "title_indices.pkl"]

# ---------------------- CSS ----------------------
st.markdown("""
<style>
body { background:#f5f7fa; }
.header { font-size:28px; font-weight:800; color:#111; }
.sub { color:#555; margin-bottom:16px; }
.card {
    background:white;
    border:1px solid #ddd;
    border-radius:10px;
    padding:14px;
    margin-bottom:12px;
    box-shadow:0 2px 12px rgba(0,0,0,0.06);
}
.movie-title { font-size:17px; font-weight:700; color:#111; }
.movie-meta { color:#555; font-size:13px; margin-bottom:6px; }
.badge {
    background:#0099ff;
    color:white;
    padding:6px 12px;
    border-radius:20px;
    font-weight:700;
}
</style>
""", unsafe_allow_html=True)

# ---------------------- Helper Functions ----------------------

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


def get_tmdb_poster(api_key, title, year=None):
    try:
        params = {"api_key": api_key, "query": title}
        if year:
            params["year"] = year
        r = requests.get("https://api.themoviedb.org/3/search/movie", params=params, timeout=8)
        data = r.json().get("results", [])
        if not data:
            return None
        poster = data[0].get("poster_path")
        if poster:
            return f"https://image.tmdb.org/t/p/w300{poster}"
    except Exception:
        return None
    return None


@st.cache_data(ttl=24*3600)
def load_image(url):
    try:
        r = requests.get(url, timeout=8)
        return Image.open(BytesIO(r.content))
    except Exception:
        return None


@st.cache_data(ttl=24*3600)
def fetch_movie_details_tmdb(api_key, movie_id):
    try:
        r = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}",
                         params={"api_key": api_key}, timeout=8)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def get_tmdb_trailer(api_key, movie_id):
    try:
        r = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}/videos",
                         params={"api_key": api_key}, timeout=8)
        data = r.json().get("results", [])
        for v in data:
            if v.get("site") == "YouTube" and v.get("type") in ("Trailer", "Teaser"):
                return f"https://www.youtube.com/watch?v={v.get('key')}"
    except Exception:
        return None
    return None


# Fuzzy Search
try:
    from rapidfuzz import process as rf_process
    def fuzzy_choices(query, choices, limit=200):
        if not query:
            return choices[:limit]
        return [m[0] for m in rf_process.extract(query, choices, limit=limit)]
except:
    import difflib
    def fuzzy_choices(query, choices, limit=200):
        if not query:
            return choices[:limit]
        return difflib.get_close_matches(query, choices, n=limit, cutoff=0.1)


# ---------------------- Load Models ----------------------

ok = check_models(MODEL_DIR)

if ok:
    tfidf, cosine_sim, md, indices, err = load_models(MODEL_DIR)
    if err:
        ok = False

if not ok:
    st.error("❌ Model files missing or error in loading!")
    if err:
        st.code(err)
    st.stop()

if "title" not in md.columns:
    st.error("Dataset missing 'title' column.")
    st.stop()

# ---------------------- SIDEBAR ----------------------

tmdb_key = st.sidebar.text_input("TMDB API key (optional)", type="password")
show_posters = st.sidebar.checkbox("Show posters", value=False)

top_default = st.sidebar.slider("Default recommendations count", 5, 25, 10)

# CSV Upload Section
st.sidebar.markdown("## 📁 Upload any CSV")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
uploaded_df = None

if uploaded_file:
    uploaded_df = pd.read_csv(uploaded_file)
    st.sidebar.success("CSV Uploaded Successfully!")

# ---------------------- Header ----------------------

st.markdown("<div class='header'>🎬 Movie Recommender</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>Light, clean UI • Genre + Overview based recommendations</div>", unsafe_allow_html=True)

# ---------------------- Search ----------------------

titles = sorted(md["title"].dropna().unique().tolist())

st.markdown("### Search & Recommend")
filter_text = st.text_input("Type to filter:")
filtered = fuzzy_choices(filter_text, titles, limit=500) if filter_text else titles[:500]
selected = st.selectbox("Select a movie:", ["-- choose --"] + filtered)

topn = st.slider("Number of recommendations", 5, 30, top_default)

button = st.button("Recommend")

# ---------------------- Recommendation Logic ----------------------

def recommend(title, topn=10):
    if title in indices.index:
        idx = indices[title]
    else:
        match = md[md["title"].str.contains(title, case=False, na=False)]
        if match.empty:
            return None
        idx = match.index[0]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:topn+1]

    movie_indexes = [i[0] for i in sim_scores]
    result = md.iloc[movie_indexes][["title", "release_date", "id"]].copy()
    result["score"] = [round(i[1], 4) for i in sim_scores]

    return result.reset_index(drop=True)

# ---------------------- Show Recommendation Results ----------------------

if button:
    if selected == "-- choose --":
        st.warning("Please select a valid movie!")
    else:
        results = recommend(selected, topn)

        if results is None:
            st.warning("No recommendations found.")
        else:
            st.markdown(f"### Top {len(results)} Recommendations for **{selected}**")

            # Graph of Score
            chart_df = results[["title", "score"]].copy()
            chart = alt.Chart(chart_df).mark_bar().encode(
                x="title:N",
                y="score:Q",
                tooltip=["title", "score"]
            ).properties(height=300)

            st.altair_chart(chart, use_container_width=True)

            st.dataframe(results)

# ---------------------- CSV ANALYSIS SECTION ----------------------

if uploaded_df is not None:
    st.markdown("## 📊 CSV Analysis & Visualization")

    st.write("### 🔍 First 10 rows")
    st.dataframe(uploaded_df.head(10))

    numeric_cols = uploaded_df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if numeric_cols:
        col = st.selectbox("Choose numeric column for graph:", numeric_cols)
        graph_type = st.radio("Graph Type:", ["Bar Chart", "Line Chart", "Area Chart"])

        df_chart = uploaded_df.reset_index().rename(columns={"index": "Row"})

        if graph_type == "Bar Chart":
            chart = alt.Chart(df_chart).mark_bar().encode(
                x="Row:O", y=f"{col}:Q", tooltip=[col]
            )
        elif graph_type == "Line Chart":
            chart = alt.Chart(df_chart).mark_line(point=True).encode(
                x="Row:O", y=f"{col}:Q", tooltip=[col]
            )
        else:
            chart = alt.Chart(df_chart).mark_area().encode(
                x="Row:O", y=f"{col}:Q", tooltip=[col]
            )

        st.altair_chart(chart, use_container_width=True)

        st.markdown("### 📌 Statistics")
        st.write(uploaded_df[col].describe())


# ---------------------- END ----------------------
