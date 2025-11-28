# app.py
# Movie Recommender — Clean final: fuzzy search, CSV upload, CSV export, trailer embed, charts (NO favorites)

import os
import traceback
import streamlit as st
import pandas as pd
import joblib
import requests
from PIL import Image
from io import BytesIO
import altair as alt

# Page config
st.set_page_config(page_title="Movie Recommender", layout="wide", initial_sidebar_state="expanded")

# --- Config ---
MODEL_DIR = "./recommender_models"
REQUIRED = [
    "tfidf_vectorizer.joblib",
    "cosine_sim.joblib",
    "movies_metadata_df.pkl",
    "title_indices.pkl",
]
SAMPLE_IMAGE = "3cf9b069-013d-4d9e-82f3-e747a5930def.png"

# --- Minimal CSS for cards ---
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

# ---------- HELPERS ----------
def check_models(path: str) -> bool:
    """Return True if model folder exists and all REQUIRED files are present."""
    return os.path.isdir(path) and all(os.path.exists(os.path.join(path, f)) for f in REQUIRED)


def load_models(path: str):
    """Try to load model artifacts. Returns (tfidf, cosine_sim, md, indices, error_str)"""
    try:
        tfidf = joblib.load(os.path.join(path, "tfidf_vectorizer.joblib"))
        cosine_sim = joblib.load(os.path.join(path, "cosine_sim.joblib"))
        md = pd.read_pickle(os.path.join(path, "movies_metadata_df.pkl"))
        indices = pd.read_pickle(os.path.join(path, "title_indices.pkl"))
        return tfidf, cosine_sim, md, indices, None
    except Exception:
        return None, None, None, None, traceback.format_exc()


def get_tmdb_poster(api_key: str, title: str, year: str | None = None) -> str | None:
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


@st.cache_data(ttl=24 * 3600)
def load_image(url: str):
    try:
        r = requests.get(url, timeout=8)
        img = Image.open(BytesIO(r.content))
        return img
    except Exception:
        return None


@st.cache_data(ttl=24 * 3600)
def fetch_movie_details_tmdb(api_key: str, movie_id: int):
    try:
        r = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}", params={"api_key": api_key}, timeout=8)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None


def get_tmdb_trailer(api_key: str, movie_id: int) -> str | None:
    try:
        r = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}/videos", params={"api_key": api_key}, timeout=8)
        if r.status_code != 200:
            return None
        data = r.json().get("results", [])
        for v in data:
            if v.get("site") == "YouTube" and v.get("type") in ("Trailer", "Teaser"):
                return f"https://www.youtube.com/watch?v={v.get('key')}"
    except Exception:
        return None
    return None


# ---------- FUZZY SEARCH ----------
try:
    from rapidfuzz import process as rf_process

    def fuzzy_choices(query, choices, limit=200):
        if not query:
            return choices[:limit]
        matches = rf_process.extract(query, choices, limit=limit)
        return [m[0] for m in matches]

except Exception:
    import difflib

    def fuzzy_choices(query, choices, limit=200):
        if not query:
            return choices[:limit]
        return difflib.get_close_matches(query, choices, n=limit, cutoff=0.1)


# ---------- LOAD MODELS ----------
ok = check_models(MODEL_DIR)
tfidf = cosine_sim = md = indices = None
err = None

if ok:
    tfidf, cosine_sim, md, indices, err = load_models(MODEL_DIR)
    if err:
        ok = False

# If models available, ensure titles column exists
if ok and "title" not in md.columns:
    st.error("movies_metadata_df.pkl missing required column: 'title'")
    st.stop()

# ============================================
# SIDEBAR: API key, options, CSV uploader (single place only)
# ============================================
st.sidebar.title("⚙️ Options")
tmdb_key = st.sidebar.text_input("TMDB API key (optional)", type="password", key="tmdb_key")
show_posters = st.sidebar.checkbox("Show posters", value=False)
top_default = st.sidebar.slider("Default recommendations count", 5, 25, 10)

st.sidebar.markdown("### 📁 Upload any CSV")
uploaded_csv = st.sidebar.file_uploader(
    "Drag and drop file here",
    type=["csv"],
    accept_multiple_files=False,
    help="Limit 200MB per file • CSV",
    key="uploaded_csv",
)

uploaded_df = None
if uploaded_csv is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_csv)
        st.sidebar.success(f"CSV loaded — {len(uploaded_df)} rows")
    except Exception as e:
        st.sidebar.error("Error reading CSV. Make sure file is a valid CSV.")
        st.sidebar.code(str(e))

if uploaded_df is not None:
    # Display a small preview under sidebar (and provide download)
    st.sidebar.markdown("Preview (first 5 rows)")
    st.sidebar.dataframe(uploaded_df.head(5))
    csv_bytes = uploaded_df.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button("Download uploaded CSV", data=csv_bytes, file_name="uploaded_file.csv", mime="text/csv")

st.sidebar.markdown("---")

# ---------- HEADER ----------
colA, colB = st.columns([4, 1])
with colA:
    st.markdown("<div class='header'>🎬 Movie Recommender</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub'>Light, clean UI • Genre + Overview based recommendations</div>", unsafe_allow_html=True)
with colB:
    if os.path.exists(SAMPLE_IMAGE):
        st.image(SAMPLE_IMAGE, width=180)

# ---------- IF MODELS MISSING, show helpful message and stop ----------
if not ok:
    st.error("Model files missing or cannot be loaded from the repository's `recommender_models` folder.")
    st.info(
        "Make sure you uploaded the `recommender_models` folder with the required files and `requirements.txt` contains all dependencies."
    )
    if err:
        st.code(err)
    st.stop()

# ---------- TITLE LIST ----------
titles = sorted(md["title"].dropna().unique().tolist())

# ---------- SEARCH UI ----------
st.markdown("### Search & Recommend")

col1, col2 = st.columns([3, 1])
with col1:
    filter_text = st.text_input("Type to filter:", value="", key="filter_text")
    filtered = fuzzy_choices(filter_text, titles, limit=500) if filter_text.strip() else titles[:500]
    if not filtered:
        filtered = ["No match found"]

    selected = st.selectbox("Select a movie:", ["-- choose --"] + filtered, key="select_movie")

    topn = st.slider("Number of recommendations", 5, 30, top_default, key="topn")
    button = st.button("Recommend", key="recommend_button")

with col2:
    st.markdown("#### How it works")
    st.write("- TF-IDF vectorization")
    st.write("- Cosine Similarity comparison")
    st.write("- Higher score = more similar")

# ---------- RECOMMENDER ----------
def recommend(title, topn=10):
    if title in indices.index:
        idx = indices[title]
    else:
        df = md[md["title"].str.contains(title, case=False, na=False)]
        if df.empty:
            return None
        idx = df.index[0]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1 : topn + 1]

    movie_indexes = [i[0] for i in sim_scores]
    result = md.iloc[movie_indexes][["title", "release_date", "id"]].copy()
    result["score"] = [round(i[1], 4) for i in sim_scores]
    return result.reset_index(drop=True)


# ---------- DISPLAY RESULTS ----------
if button:
    if selected == "-- choose --" or selected == "No match found":
        st.warning("Please select a valid movie.")
    else:
        results = recommend(selected, topn)
        if results is None or results.empty:
            st.warning("No recommendations found.")
        else:
            # CSV download
            csv = results.to_csv(index=False).encode("utf-8")
            st.download_button("Download recommendations (CSV)", data=csv, file_name=f"reco_{selected}.csv", mime="text/csv")

            # Optionally fetch popularity from TMDB if key provided
            popularity_available = False
            if tmdb_key:
                pops = []
                for mid in results["id"]:
                    try:
                        md_tmdb = fetch_movie_details_tmdb(tmdb_key, int(mid)) if str(mid).isdigit() else None
                    except Exception:
                        md_tmdb = None
                    pops.append(md_tmdb.get("popularity", 0) if md_tmdb else 0)

                results = results.copy()
                results["popularity"] = pops
                if any(p > 0 for p in pops):
                    popularity_available = True

            left, right = st.columns([2.5, 1])

            # LEFT: charts & cards
            with left:
                st.markdown(f"### Top {len(results)} recommendations for **{selected}**")

                # Score bar chart
                try:
                    chart_df = results[["title", "score"]].copy()
                    chart = (
                        alt.Chart(chart_df)
                        .mark_bar()
                        .encode(x=alt.X("title:N", sort="-y", title="Movie"), y=alt.Y("score:Q", title="Similarity Score"), tooltip=["title", "score"])
                        .properties(height=320)
                    )
                    st.altair_chart(chart, use_container_width=True)
                except Exception:
                    st.write("Could not render score chart.")

                # Popularity chart
                if popularity_available:
                    try:
                        p_df = results[["title", "popularity"]].copy()
                        p_chart = (
                            alt.Chart(p_df)
                            .mark_line(point=True)
                            .encode(x=alt.X("title:N", sort="-y", title="Movie"), y=alt.Y("popularity:Q", title="TMDB Popularity"))
                            .properties(height=240)
                        )
                        st.altair_chart(p_chart, use_container_width=True)
                    except Exception:
                        st.write("Could not render popularity chart.")

                # Recommendation cards (no favorites)
                for i, r in results.iterrows():
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='movie-title'>{i+1}. {r['title']}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='movie-meta'>{r.get('release_date', '')} • id: {r.get('id','')}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='badge'>{r['score']}</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

            # RIGHT: table, poster, trailer
            with right:
                st.markdown("#### Table view")
                st.dataframe(results)

                # Poster for top result if requested
                if show_posters and tmdb_key:
                    try:
                        top_title = results.iloc[0]["title"]
                        year = None
                        rd = results.iloc[0].get("release_date", "")
                        if isinstance(rd, str) and len(rd) >= 4:
                            year = rd[:4]
                        poster_url = get_tmdb_poster(tmdb_key, top_title, year)
                        if poster_url:
                            img = load_image(poster_url)
                            if img:
                                st.image(img, caption=top_title, use_column_width=True)
                            else:
                                st.info("Poster not available.")
                        else:
                            st.info("Poster not found.")
                    except Exception:
                        pass

                # Trailer embed
                if tmdb_key:
                    try:
                        top_id = int(results.iloc[0]["id"])
                        trailer = get_tmdb_trailer(tmdb_key, top_id)
                        if trailer:
                            st.markdown("**Trailer:**")
                            st.video(trailer)
                    except Exception:
                        pass

# ---------- If user uploaded CSV and wants to use it ----------
# For quick testing: show uploaded CSV in main area when uploaded and no recommend action
if uploaded_df is not None and not button:
    st.markdown("### Uploaded CSV (main view)")
    st.dataframe(uploaded_df)
    st.info("You uploaded a CSV — use it to inspect columns or download it from the sidebar.")

# ---------- END ----------
# Notes:
# - Dependencies (requirements.txt): streamlit, pandas, numpy, scikit-learn, joblib, requests, pillow, altair, rapidfuzz (optional)
