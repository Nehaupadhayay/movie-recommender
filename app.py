# app.py
# Movie Recommender — Integrated: fuzzy search, favorites, CSV export, trailer embed, graphs

import os
import traceback
import streamlit as st
import pandas as pd# app.py
# Movie Recommender — Integrated: fuzzy search, favorites, CSV export, trailer embed, graphs
# app.py
# Movie Recommender — Integrated: fuzzy search, favorites, CSV export, trailer embed, graphs

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
REQUIRED = ["tfidf_vectorizer.joblib", "cosine_sim.joblib", "movies_metadata_df.pkl", "title_indices.pkl"]

SAMPLE_IMAGE = "3cf9b069-013d-4d9e-82f3-e747a5930def.png"

# ---------- LIGHT CSS ----------
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
.btn-small { padding:6px 8px; font-size:13px; }
</style>
""", unsafe_allow_html=True)

# ---------- HELPERS ----------

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
        img = Image.open(BytesIO(r.content))
        return img
    except Exception:
        return None

@st.cache_data(ttl=24*3600)
def fetch_movie_details_tmdb(api_key, movie_id):
    try:
        r = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}", params={"api_key": api_key}, timeout=8)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None

def get_tmdb_trailer(api_key, movie_id):
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

# ---------- ENHANCED SEARCH: fuzzy / autocomplete ----------
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

# ensure md has necessary columns
if ok:
    if "title" not in md.columns:
        st.error("movies_metadata_df.pkl missing 'title' column")
        st.stop()

# ---------- session_state init ----------
if "favorites" not in st.session_state:
    st.session_state["favorites"] = []

# helper callbacks
def add_favorite_cb(title):
    if title and title not in st.session_state["favorites"]:
        st.session_state["favorites"].append(title)
    # rerun so sidebar updates immediately
    st.experimental_rerun()

def remove_favorite_cb(idx):
    try:
        st.session_state["favorites"].pop(idx)
    except Exception:
        pass
    st.experimental_rerun()

# ---------- SIDEBAR ----------
tmdb_key = st.sidebar.text_input("TMDB API key (optional)", type="password")
show_posters = st.sidebar.checkbox("Show posters", value=False)
top_default = st.sidebar.slider("Default recommendations count", 5, 25, 10)

with st.sidebar.expander("⭐ Favorites", expanded=False):
    favs = st.session_state["favorites"]
    if favs:
        for fi, mv in enumerate(favs):
            col_a, col_b = st.columns([4,1])
            with col_a:
                st.write(f"{fi+1}. {mv}")
            with col_b:
                # remove button for each favorite
                if st.button("Remove", key=f"fav_rem_{fi}"):
                    remove_favorite_cb(fi)
        st.markdown("---")
        # download favorites CSV
        try:
            fav_df = pd.DataFrame({"title": favs})
            csv = fav_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download favorites (CSV)", data=csv, file_name="favorites.csv", mime="text/csv")
        except Exception:
            pass
        if st.button("Clear favorites", key="fav_clear"):
            st.session_state["favorites"].clear()
            st.experimental_rerun()
    else:
        st.write("No favorites yet.")

# ---------- HEADER ----------
colA, colB = st.columns([4,1])
with colA:
    st.markdown("<div class='header'>🎬 Movie Recommender</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub'>Light, clean UI • Genre + Overview based recommendations</div>", unsafe_allow_html=True)
with colB:
    if os.path.exists(SAMPLE_IMAGE):
        st.image(SAMPLE_IMAGE, width=200)

# ---------- ERROR IF MODELS MISSING ----------
if not ok:
    st.error("Model files missing or cannot be loaded.")
    if err:
        st.code(err)
    st.stop()

# ---------- TITLE LIST ----------
titles = md["title"].dropna().unique().tolist()
titles = sorted(titles)

# ---------- SEARCH UI ----------
st.markdown("### Search & Recommend")

col1, col2 = st.columns([3,1])
with col1:
    filter_text = st.text_input("Type to filter:", "")
    if filter_text.strip():
        filtered = fuzzy_choices(filter_text, titles, limit=500)
        if not filtered:
            filtered = ["No match found"]
    else:
        filtered = titles[:500]

    selected = st.selectbox("Select a movie:", ["-- choose --"] + filtered)

    topn = st.slider("Number of recommendations", 5, 30, top_default)

    button = st.button("Recommend")

with col2:
    st.markdown("#### How it works")
    st.write("- Builds text = genres + overview")
    st.write("- Uses TF-IDF to convert text → vector")
    st.write("- Uses Cosine Similarity to find closest vectors")
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
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:topn+1]

    movie_indexes = [i[0] for i in sim_scores]
    result = md.iloc[movie_indexes][["title", "release_date", "id"]].copy()
    result["score"] = [round(i[1], 4) for i in sim_scores]
    return result.reset_index(drop=True)

# ---------- SHOW RESULTS & GRAPHS ----------
if button:
    if selected == "-- choose --" or selected == "No match found":
        st.warning("Please select a valid movie.")
    else:
        results = recommend(selected, topn)
        if results is None:
            st.warning("No recommendations found.")
        else:
            # Export button (CSV)
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button("Download recommendations (CSV)", data=csv, file_name=f"reco_{selected}.csv", mime="text/csv")

            # Try to fetch popularity if TMDB key provided and ids look numeric
            popularity_available = False
            if tmdb_key:
                pops = []
                for mid in results["id"]:
                    try:
                        md_tmdb = fetch_movie_details_tmdb(tmdb_key, int(mid))
                    except Exception:
                        md_tmdb = None
                    if md_tmdb and "popularity" in md_tmdb:
                        pops.append(md_tmdb.get("popularity", 0))
                    else:
                        pops.append(0)
                results = results.copy()
                results["popularity"] = pops
                if any([p > 0 for p in pops]):
                    popularity_available = True

            left, right = st.columns([2.5,1])
            with left:
                st.markdown(f"### Top {len(results)} recommendations for **{selected}**")

                # Show a bar chart of scores (Altair)
                try:
                    chart_df = results[["title", "score"]].copy()
                    chart_df["title"] = chart_df["title"].astype(str)
                    chart = alt.Chart(chart_df).mark_bar().encode(
                        x=alt.X('title:N', sort='-y', title='Movie'),
                        y=alt.Y('score:Q', title='Similarity Score'),
                        tooltip=['title', 'score']
                    ).properties(width='container', height=320)
                    st.altair_chart(chart, use_container_width=True)
                except Exception:
                    st.write("Could not render score chart.")

                # If popularity was fetched, show a second chart
                if popularity_available:
                    try:
                        pop_df = results[["title", "popularity"]].copy()
                        pop_df["title"] = pop_df["title"].astype(str)
                        pop_chart = alt.Chart(pop_df).mark_line(point=True).encode(
                            x=alt.X('title:N', sort='-y', title='Movie'),
                            y=alt.Y('popularity:Q', title='TMDB Popularity'),
                            tooltip=['title', 'popularity']
                        ).properties(width='container', height=240)
                        st.altair_chart(pop_chart, use_container_width=True)
                    except Exception:
                        st.write("Could not render popularity chart.")

                # List results as cards with add-to-favorites button
                for i, r in results.iterrows():
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    c1, c2 = st.columns([8,1])
                    with c1:
                        st.markdown(f"<div class='movie-title'>{i+1}. {r['title']}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='movie-meta'>{r['release_date']} • id: {r['id']}</div>", unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"<div class='badge'>{r['score']}</div>", unsafe_allow_html=True)
                    # favorites button
                    fav_key = f"fav_add_{selected}_{i}"
                    if st.button("➕ Add to favorites", key=fav_key):
                        add_favorite_cb(r["title"])
                    st.markdown("</div>", unsafe_allow_html=True)

            with right:
                st.markdown("#### Table view")
                st.dataframe(results)

                # show poster for top result
                if show_posters and tmdb_key:
                    top_title = results.iloc[0]["title"]
                    year = None
                    if isinstance(results.iloc[0]["release_date"], str) and len(results.iloc[0]["release_date"]) >= 4:
                        year = results.iloc[0]["release_date"][:4]

                    poster_url = get_tmdb_poster(tmdb_key, top_title, year)
                    if poster_url:
                        img = load_image(poster_url)
                        if img:
                            st.image(img, caption=top_title, use_column_width=True)
                        else:
                            st.info("Poster not available.")
                    else:
                        st.info("Poster not found.")

                # embed trailer if available
                if tmdb_key:
                    try:
                        top_id = int(results.iloc[0]["id"])
                        trailer = get_tmdb_trailer(tmdb_key, top_id)
                        if trailer:
                            st.markdown("**Trailer:**")
                            st.video(trailer)
                    except Exception:
                        pass

# ---------- END ----------

# Notes:
# - Integrated features: fuzzy search (rapidfuzz/difflib), favorites (session_state), CSV export, trailer embed, score & popularity charts.
# - Run with: `streamlit run app.py`
# - Dependencies (requirements.txt): streamlit, pandas, joblib, requests, pillow, altair, rapidfuzz (optional)
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
REQUIRED = ["tfidf_vectorizer.joblib", "cosine_sim.joblib", "movies_metadata_df.pkl", "title_indices.pkl"]

SAMPLE_IMAGE = "3cf9b069-013d-4d9e-82f3-e747a5930def.png"

# ---------- LIGHT CSS ----------
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
.btn-small { padding:6px 8px; font-size:13px; }
</style>
""", unsafe_allow_html=True)

# ---------- HELPERS ----------

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
        img = Image.open(BytesIO(r.content))
        return img
    except Exception:
        return None

@st.cache_data(ttl=24*3600)
def fetch_movie_details_tmdb(api_key, movie_id):
    try:
        r = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}", params={"api_key": api_key}, timeout=8)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None

def get_tmdb_trailer(api_key, movie_id):
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

# ---------- ENHANCED SEARCH: fuzzy / autocomplete ----------
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

# ensure md has necessary columns
if ok:
    if "title" not in md.columns:
        st.error("movies_metadata_df.pkl missing 'title' column")
        st.stop()

# ---------- session_state init ----------
if "favorites" not in st.session_state:
    st.session_state["favorites"] = []

# helper callbacks
def add_favorite_cb(title):
    if title and title not in st.session_state["favorites"]:
        st.session_state["favorites"].append(title)
    # rerun so sidebar updates immediately
    st.experimental_rerun()

def remove_favorite_cb(idx):
    try:
        st.session_state["favorites"].pop(idx)
    except Exception:
        pass
    st.experimental_rerun()

# ---------- SIDEBAR ----------
tmdb_key = st.sidebar.text_input("TMDB API key (optional)", type="password")
show_posters = st.sidebar.checkbox("Show posters", value=False)
top_default = st.sidebar.slider("Default recommendations count", 5, 25, 10)

with st.sidebar.expander("⭐ Favorites", expanded=False):
    favs = st.session_state["favorites"]
    if favs:
        for fi, mv in enumerate(favs):
            col_a, col_b = st.columns([4,1])
            with col_a:
                st.write(f"{fi+1}. {mv}")
            with col_b:
                # remove button for each favorite
                if st.button("Remove", key=f"fav_rem_{fi}"):
                    remove_favorite_cb(fi)
        st.markdown("---")
        # download favorites CSV
        try:
            fav_df = pd.DataFrame({"title": favs})
            csv = fav_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download favorites (CSV)", data=csv, file_name="favorites.csv", mime="text/csv")
        except Exception:
            pass
        if st.button("Clear favorites", key="fav_clear"):
            st.session_state["favorites"].clear()
            st.experimental_rerun()
    else:
        st.write("No favorites yet.")

# ---------- HEADER ----------
colA, colB = st.columns([4,1])
with colA:
    st.markdown("<div class='header'>🎬 Movie Recommender</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub'>Light, clean UI • Genre + Overview based recommendations</div>", unsafe_allow_html=True)
with colB:
    if os.path.exists(SAMPLE_IMAGE):
        st.image(SAMPLE_IMAGE, width=200)

# ---------- ERROR IF MODELS MISSING ----------
if not ok:
    st.error("Model files missing or cannot be loaded.")
    if err:
        st.code(err)
    st.stop()

# ---------- TITLE LIST ----------
titles = md["title"].dropna().unique().tolist()
titles = sorted(titles)

# ---------- SEARCH UI ----------
st.markdown("### Search & Recommend")

col1, col2 = st.columns([3,1])
with col1:
    filter_text = st.text_input("Type to filter:", "")
    if filter_text.strip():
        filtered = fuzzy_choices(filter_text, titles, limit=500)
        if not filtered:
            filtered = ["No match found"]
    else:
        filtered = titles[:500]

    selected = st.selectbox("Select a movie:", ["-- choose --"] + filtered)

    topn = st.slider("Number of recommendations", 5, 30, top_default)

    button = st.button("Recommend")

with col2:
    st.markdown("#### How it works")
    st.write("- Builds text = genres + overview")
    st.write("- Uses TF-IDF to convert text → vector")
    st.write("- Uses Cosine Similarity to find closest vectors")
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
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:topn+1]

    movie_indexes = [i[0] for i in sim_scores]
    result = md.iloc[movie_indexes][["title", "release_date", "id"]].copy()
    result["score"] = [round(i[1], 4) for i in sim_scores]
    return result.reset_index(drop=True)

# ---------- SHOW RESULTS & GRAPHS ----------
if button:
    if selected == "-- choose --" or selected == "No match found":
        st.warning("Please select a valid movie.")
    else:
        results = recommend(selected, topn)
        if results is None:
            st.warning("No recommendations found.")
        else:
            # Export button (CSV)
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button("Download recommendations (CSV)", data=csv, file_name=f"reco_{selected}.csv", mime="text/csv")

            # Try to fetch popularity if TMDB key provided and ids look numeric
            popularity_available = False
            if tmdb_key:
                pops = []
                for mid in results["id"]:
                    try:
                        md_tmdb = fetch_movie_details_tmdb(tmdb_key, int(mid))
                    except Exception:
                        md_tmdb = None
                    if md_tmdb and "popularity" in md_tmdb:
                        pops.append(md_tmdb.get("popularity", 0))
                    else:
                        pops.append(0)
                results = results.copy()
                results["popularity"] = pops
                if any([p > 0 for p in pops]):
                    popularity_available = True

            left, right = st.columns([2.5,1])
            with left:
                st.markdown(f"### Top {len(results)} recommendations for **{selected}**")

                # Show a bar chart of scores (Altair)
                try:
                    chart_df = results[["title", "score"]].copy()
                    chart_df["title"] = chart_df["title"].astype(str)
                    chart = alt.Chart(chart_df).mark_bar().encode(
                        x=alt.X('title:N', sort='-y', title='Movie'),
                        y=alt.Y('score:Q', title='Similarity Score'),
                        tooltip=['title', 'score']
                    ).properties(width='container', height=320)
                    st.altair_chart(chart, use_container_width=True)
                except Exception:
                    st.write("Could not render score chart.")

                # If popularity was fetched, show a second chart
                if popularity_available:
                    try:
                        pop_df = results[["title", "popularity"]].copy()
                        pop_df["title"] = pop_df["title"].astype(str)
                        pop_chart = alt.Chart(pop_df).mark_line(point=True).encode(
                            x=alt.X('title:N', sort='-y', title='Movie'),
                            y=alt.Y('popularity:Q', title='TMDB Popularity'),
                            tooltip=['title', 'popularity']
                        ).properties(width='container', height=240)
                        st.altair_chart(pop_chart, use_container_width=True)
                    except Exception:
                        st.write("Could not render popularity chart.")

                # List results as cards with add-to-favorites button
                for i, r in results.iterrows():
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    c1, c2 = st.columns([8,1])
                    with c1:
                        st.markdown(f"<div class='movie-title'>{i+1}. {r['title']}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='movie-meta'>{r['release_date']} • id: {r['id']}</div>", unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"<div class='badge'>{r['score']}</div>", unsafe_allow_html=True)
                    # favorites button
                    fav_key = f"fav_add_{selected}_{i}"
                    if st.button("➕ Add to favorites", key=fav_key):
                        add_favorite_cb(r["title"])
                    st.markdown("</div>", unsafe_allow_html=True)

            with right:
                st.markdown("#### Table view")
                st.dataframe(results)

                # show poster for top result
                if show_posters and tmdb_key:
                    top_title = results.iloc[0]["title"]
                    year = None
                    if isinstance(results.iloc[0]["release_date"], str) and len(results.iloc[0]["release_date"]) >= 4:
                        year = results.iloc[0]["release_date"][:4]

                    poster_url = get_tmdb_poster(tmdb_key, top_title, year)
                    if poster_url:
                        img = load_image(poster_url)
                        if img:
                            st.image(img, caption=top_title, use_column_width=True)
                        else:
                            st.info("Poster not available.")
                    else:
                        st.info("Poster not found.")

                # embed trailer if available
                if tmdb_key:
                    try:
                        top_id = int(results.iloc[0]["id"])
                        trailer = get_tmdb_trailer(tmdb_key, top_id)
                        if trailer:
                            st.markdown("**Trailer:**")
                            st.video(trailer)
                    except Exception:
                        pass

# ---------- END ----------

# Notes:
# - Integrated features: fuzzy search (rapidfuzz/difflib), favorites (session_state), CSV export, trailer embed, score & popularity charts.
# - Run with: `streamlit run app.py`
# - Dependencies (requirements.txt): streamlit, pandas, joblib, requests, pillow, altair, rapidfuzz (optional)
import joblib
import requests
from PIL import Image
from io import BytesIO
import altair as alt

st.set_page_config(page_title="Movie Recommender", layout="wide")

MODEL_DIR = "./recommender_models"
REQUIRED = ["tfidf_vectorizer.joblib", "cosine_sim.joblib", "movies_metadata_df.pkl", "title_indices.pkl"]

SAMPLE_IMAGE = "3cf9b069-013d-4d9e-82f3-e747a5930def.png"

# ---------- LIGHT CSS ----------
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
.btn-small { padding:6px 8px; font-size:13px; }
</style>
""", unsafe_allow_html=True)

# ---------- HELPERS ----------

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
        img = Image.open(BytesIO(r.content))
        return img
    except Exception:
        return None

@st.cache_data(ttl=24*3600)
def fetch_movie_details_tmdb(api_key, movie_id):
    try:
        r = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}", params={"api_key": api_key}, timeout=8)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None

def get_tmdb_trailer(api_key, movie_id):
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

# ---------- ENHANCED SEARCH: fuzzy / autocomplete ----------
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

# ensure md has necessary columns
if ok:
    if "title" not in md.columns:
        st.error("movies_metadata_df.pkl missing 'title' column")
        st.stop()

# ---------- session_state init ----------
if "favorites" not in st.session_state:
    st.session_state["favorites"] = []

# helper callbacks
def add_favorite_cb(title):
    if title and title not in st.session_state["favorites"]:
        st.session_state["favorites"].append(title)
    # rerun so sidebar updates immediately
    st.experimental_rerun()

def remove_favorite_cb(idx):
    try:
        st.session_state["favorites"].pop(idx)
    except Exception:
        pass
    st.experimental_rerun()

# ---------- SIDEBAR ----------
tmdb_key = st.sidebar.text_input("TMDB API key (optional)", type="password")
show_posters = st.sidebar.checkbox("Show posters", value=False)
top_default = st.sidebar.slider("Default recommendations count", 5, 25, 10)

with st.sidebar.expander("⭐ Favorites", expanded=False):
    favs = st.session_state["favorites"]
    if favs:
        for fi, mv in enumerate(favs):
            col_a, col_b = st.columns([4,1])
            with col_a:
                st.write(f"{fi+1}. {mv}")
            with col_b:
                # remove button for each favorite
                if st.button("Remove", key=f"fav_rem_{fi}"):
                    remove_favorite_cb(fi)
        st.markdown("---")
        # download favorites CSV
        try:
            fav_df = pd.DataFrame({"title": favs})
            csv = fav_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download favorites (CSV)", data=csv, file_name="favorites.csv", mime="text/csv")
        except Exception:
            pass
        if st.button("Clear favorites", key="fav_clear"):
            st.session_state["favorites"].clear()
            st.experimental_rerun()
    else:
        st.write("No favorites yet.")

# ---------- HEADER ----------
colA, colB = st.columns([4,1])
with colA:
    st.markdown("<div class='header'>🎬 Movie Recommender</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub'>Light, clean UI • Genre + Overview based recommendations</div>", unsafe_allow_html=True)
with colB:
    if os.path.exists(SAMPLE_IMAGE):
        st.image(SAMPLE_IMAGE, width=200)

# ---------- ERROR IF MODELS MISSING ----------
if not ok:
    st.error("Model files missing or cannot be loaded.")
    if err:
        st.code(err)
    st.stop()

# ---------- TITLE LIST ----------
titles = md["title"].dropna().unique().tolist()
titles = sorted(titles)

# ---------- SEARCH UI ----------
st.markdown("### Search & Recommend")

col1, col2 = st.columns([3,1])
with col1:
    filter_text = st.text_input("Type to filter:", "")
    if filter_text.strip():
        filtered = fuzzy_choices(filter_text, titles, limit=500)
        if not filtered:
            filtered = ["No match found"]
    else:
        filtered = titles[:500]

    selected = st.selectbox("Select a movie:", ["-- choose --"] + filtered)

    topn = st.slider("Number of recommendations", 5, 30, top_default)

    button = st.button("Recommend")

with col2:
    st.markdown("#### How it works")
    st.write("- Builds text = genres + overview")
    st.write("- Uses TF-IDF to convert text → vector")
    st.write("- Uses Cosine Similarity to find closest vectors")
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
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:topn+1]

    movie_indexes = [i[0] for i in sim_scores]
    result = md.iloc[movie_indexes][["title", "release_date", "id"]].copy()
    result["score"] = [round(i[1], 4) for i in sim_scores]
    return result.reset_index(drop=True)

# ---------- SHOW RESULTS & GRAPHS ----------
if button:
    if selected == "-- choose --" or selected == "No match found":
        st.warning("Please select a valid movie.")
    else:
        results = recommend(selected, topn)
        if results is None:
            st.warning("No recommendations found.")
        else:
            # Export button (CSV)
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button("Download recommendations (CSV)", data=csv, file_name=f"reco_{selected}.csv", mime="text/csv")

            # Try to fetch popularity if TMDB key provided and ids look numeric
            popularity_available = False
            if tmdb_key:
                pops = []
                for mid in results["id"]:
                    try:
                        md_tmdb = fetch_movie_details_tmdb(tmdb_key, int(mid))
                    except Exception:
                        md_tmdb = None
                    if md_tmdb and "popularity" in md_tmdb:
                        pops.append(md_tmdb.get("popularity", 0))
                    else:
                        pops.append(0)
                results = results.copy()
                results["popularity"] = pops
                if any([p > 0 for p in pops]):
                    popularity_available = True

            left, right = st.columns([2.5,1])
            with left:
                st.markdown(f"### Top {len(results)} recommendations for **{selected}**")

                # Show a bar chart of scores (Altair)
                try:
                    chart_df = results[["title", "score"]].copy()
                    chart_df["title"] = chart_df["title"].astype(str)
                    chart = alt.Chart(chart_df).mark_bar().encode(
                        x=alt.X('title:N', sort='-y', title='Movie'),
                        y=alt.Y('score:Q', title='Similarity Score'),
                        tooltip=['title', 'score']
                    ).properties(width='container', height=320)
                    st.altair_chart(chart, use_container_width=True)
                except Exception:
                    st.write("Could not render score chart.")

                # If popularity was fetched, show a second chart
                if popularity_available:
                    try:
                        pop_df = results[["title", "popularity"]].copy()
                        pop_df["title"] = pop_df["title"].astype(str)
                        pop_chart = alt.Chart(pop_df).mark_line(point=True).encode(
                            x=alt.X('title:N', sort='-y', title='Movie'),
                            y=alt.Y('popularity:Q', title='TMDB Popularity'),
                            tooltip=['title', 'popularity']
                        ).properties(width='container', height=240)
                        st.altair_chart(pop_chart, use_container_width=True)
                    except Exception:
                        st.write("Could not render popularity chart.")

                # List results as cards with add-to-favorites button
                for i, r in results.iterrows():
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    c1, c2 = st.columns([8,1])
                    with c1:
                        st.markdown(f"<div class='movie-title'>{i+1}. {r['title']}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='movie-meta'>{r['release_date']} • id: {r['id']}</div>", unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"<div class='badge'>{r['score']}</div>", unsafe_allow_html=True)
                    # favorites button
                    fav_key = f"fav_add_{selected}_{i}"
                    if st.button("➕ Add to favorites", key=fav_key):
                        add_favorite_cb(r["title"])
                    st.markdown("</div>", unsafe_allow_html=True)

            with right:
                st.markdown("#### Table view")
                st.dataframe(results)

                # show poster for top result
                if show_posters and tmdb_key:
                    top_title = results.iloc[0]["title"]
                    year = None
                    if isinstance(results.iloc[0]["release_date"], str) and len(results.iloc[0]["release_date"]) >= 4:
                        year = results.iloc[0]["release_date"][:4]

                    poster_url = get_tmdb_poster(tmdb_key, top_title, year)
                    if poster_url:
                        img = load_image(poster_url)
                        if img:
                            st.image(img, caption=top_title, use_column_width=True)
                        else:
                            st.info("Poster not available.")
                    else:
                        st.info("Poster not found.")

                # embed trailer if available
                if tmdb_key:
                    try:
                        top_id = int(results.iloc[0]["id"])
                        trailer = get_tmdb_trailer(tmdb_key, top_id)
                        if trailer:
                            st.markdown("**Trailer:**")
                            st.video(trailer)
                    except Exception:
                        pass

# ---------- END ----------

# Notes:
# - Integrated features: fuzzy search (rapidfuzz/difflib), favorites (session_state), CSV export, trailer embed, score & popularity charts.
# - Run with: `streamlit run app.py`
# - Dependencies (requirements.txt): streamlit, pandas, joblib, requests, pillow, altair, rapidfuzz (optional)
