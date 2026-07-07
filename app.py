"""
app.py — Step 4: Interface (UX)
Music Fit Predictor — Professional Streamlit UI

Run with:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.graph_objects as go
import plotly.express as px

FEATURES = [
    "danceability", "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
    "duration_ms", "time_signature"
]

DISPLAY_FEATURES = [
    "danceability", "energy", "acousticness", "valence",
    "speechiness", "liveness", "instrumentalness"
]

# ---------------------------------------------------------------------------
# Page config & global styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Music Fit Predictor",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }

    .hero {
        background: linear-gradient(135deg, #1DB954 0%, #1ed760 50%, #169c46 100%);
        padding: 2.2rem 2rem;
        border-radius: 18px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(29, 185, 84, 0.25);
    }
    .hero h1 {
        color: #ffffff;
        font-weight: 700;
        font-size: 2.4rem;
        margin: 0;
    }
    .hero p {
        color: rgba(255,255,255,0.9);
        font-size: 1.05rem;
        margin-top: 0.4rem;
    }

    .section-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
    }

    .step-badge {
        display: inline-block;
        background: #1DB954;
        color: white;
        font-weight: 700;
        font-size: 0.8rem;
        padding: 0.25rem 0.7rem;
        border-radius: 20px;
        margin-bottom: 0.6rem;
        letter-spacing: 0.5px;
    }

    .track-chip {
        display: inline-block;
        background: rgba(29,185,84,0.15);
        border: 1px solid rgba(29,185,84,0.4);
        color: #1ed760;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.88rem;
        font-weight: 600;
    }

    .result-fit {
        background: rgba(29,185,84,0.12);
        border-left: 4px solid #1DB954;
        border-radius: 10px;
        padding: 0.85rem 1.1rem;
        margin-bottom: 0.6rem;
    }
    .result-nofit {
        background: rgba(255,82,82,0.08);
        border-left: 4px solid #ff5252;
        border-radius: 10px;
        padding: 0.85rem 1.1rem;
        margin-bottom: 0.6rem;
    }
    .result-title {
        font-weight: 700;
        font-size: 1.0rem;
        color: #f5f5f5;
    }
    .result-sub {
        color: #aaaaaa;
        font-size: 0.85rem;
    }

    .metric-pill {
        background: rgba(255,255,255,0.04);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.07);
    }
    .metric-pill .value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1ed760;
    }
    .metric-pill .label {
        color: #aaaaaa;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data & model loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/dataset.csv")
    df = df.dropna(subset=FEATURES + ["track_name", "artists"])
    df = df.drop_duplicates(subset=["track_name", "artists"]).reset_index(drop=True)
    return df


@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


df = load_data()
model = load_model()

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🎵 Music Fit Predictor</h1>
    <p>Build a taste profile from songs you love, then discover whether any track in our
    114k-song catalog matches your vibe — powered by a Random Forest classifier
    trained on Spotify audio features.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — how it works + dataset stats
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ℹ️ How it works")
    st.markdown("""
1. **Add favorites** — search and add at least 3 songs you love.
2. **Taste profile** — we compute the average audio-feature "centroid" of your favorites.
3. **Check a song** — search any track and get a Fit / Not Fit prediction with confidence.
4. **Recommendations** — see the 10 closest tracks to your taste profile.
    """)
    st.divider()
    st.markdown("### 📊 Dataset")
    st.markdown(f"""
    <div class="metric-pill" style="margin-bottom:0.6rem;">
        <div class="value">{len(df):,}</div>
        <div class="label">Tracks</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-pill">
        <div class="value">{df['track_genre'].nunique()}</div>
        <div class="label">Genres</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.caption("Model: Random Forest Classifier · Features: 13 audio attributes · Source: Spotify Tracks Dataset (Kaggle)")

# ---------------------------------------------------------------------------
# STEP 1 — Build listening history
# ---------------------------------------------------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<span class="step-badge">STEP 1</span>', unsafe_allow_html=True)
st.subheader("🎧 Your listening history")
st.write("Search for songs you love and add them to build your taste profile.")

col1, col2 = st.columns([3, 1])
with col1:
    search = st.text_input("Search a track", "", placeholder="e.g. Blinding Lights, Bohemian Rhapsody...",
                            label_visibility="collapsed")
with col2:
    if st.button("🗑️ Clear all", use_container_width=True):
        st.session_state.history = []

if search:
    results = df[df["track_name"].str.contains(search, case=False, na=False)].head(8)
    if results.empty:
        st.warning("No tracks found. Try a different search term.")
    else:
        options = (results["track_name"] + " — " + results["artists"]).tolist()
        sub1, sub2 = st.columns([3, 1])
        with sub1:
            selected = st.selectbox("Pick a track to add", options, label_visibility="collapsed")
        with sub2:
            add_clicked = st.button("➕ Add to favorites", use_container_width=True)
        if add_clicked and selected and selected not in st.session_state.history:
            st.session_state.history.append(selected)
            st.toast(f"Added: {selected.split(' — ')[0]}", icon="✅")

        with st.expander(f"View {len(results)} search results"):
            st.dataframe(
                results[["track_name", "artists", "track_genre", "popularity"]]
                .rename(columns={"track_name": "Track", "artists": "Artist",
                                  "track_genre": "Genre", "popularity": "Popularity"}),
                use_container_width=True, hide_index=True
            )

st.markdown("**Your favorites:**")
if st.session_state.history:
    chips_html = "".join(
        f'<span class="track-chip">🎵 {t}</span>' for t in st.session_state.history
    )
    st.markdown(chips_html, unsafe_allow_html=True)
    st.caption(f"{len(st.session_state.history)} track(s) added — need at least 3 to build a profile.")
else:
    st.info("No favorites yet. Search above and add at least 3 tracks to get started.")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# STEP 2, 3, 4 — only when enough favorites
# ---------------------------------------------------------------------------
if len(st.session_state.history) >= 3:
    mask = (df["track_name"] + " — " + df["artists"]).isin(st.session_state.history)
    fav_df = df[mask]
    centroid = fav_df[FEATURES].mean()

    # ---- STEP 2: Taste profile ----
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<span class="step-badge">STEP 2</span>', unsafe_allow_html=True)
    st.subheader("🧬 Your taste profile")
    st.write("This radar chart shows the average audio-feature signature of your favorite tracks.")

    radar_values = [centroid[f] for f in DISPLAY_FEATURES]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=radar_values + [radar_values[0]],
        theta=DISPLAY_FEATURES + [DISPLAY_FEATURES[0]],
        fill='toself',
        name='Your taste',
        line_color='#1DB954',
        fillcolor='rgba(29,185,84,0.25)',
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,0.1)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
            bgcolor="rgba(0,0,0,0)"
        ),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#f5f5f5",
        margin=dict(l=40, r=40, t=20, b=20),
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ---- STEP 3: Predict fit ----
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<span class="step-badge">STEP 3</span>', unsafe_allow_html=True)
    st.subheader("🔍 Check if a song fits your taste")
    check_search = st.text_input("Search a track to test", "", key="check",
                                   placeholder="Type a song name...", label_visibility="collapsed")

    if check_search:
        candidates = df[df["track_name"].str.contains(check_search, case=False, na=False)].head(8)
        if candidates.empty:
            st.warning("No tracks found. Try a different search term.")
        else:
            for _, row in candidates.iterrows():
                x = pd.DataFrame([row[FEATURES]], columns=FEATURES)
                proba = model.predict_proba(x)[0, 1]
                fits = proba >= 0.5
                css_class = "result-fit" if fits else "result-nofit"
                badge = "✅ Fits your taste" if fits else "❌ Probably not your taste"
                pct = int(proba * 100)
                bar_color = "#1DB954" if fits else "#ff5252"

                st.markdown(f"""
                <div class="{css_class}">
                    <div class="result-title">{row['track_name']} — {row['artists']}</div>
                    <div class="result-sub">{row['track_genre'].title()} · {badge}</div>
                    <div style="background:rgba(255,255,255,0.08); border-radius:6px; height:8px; margin-top:0.5rem; overflow:hidden;">
                        <div style="background:{bar_color}; width:{pct}%; height:100%;"></div>
                    </div>
                    <div class="result-sub" style="margin-top:0.3rem;">Confidence: {pct}%</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.caption("Type a track name above to get an instant Fit / Not Fit prediction.")

    st.markdown('</div>', unsafe_allow_html=True)

    # ---- STEP 4: Recommendations ----
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<span class="step-badge">STEP 4</span>', unsafe_allow_html=True)
    st.subheader("⭐ Recommended for you")
    st.write("The 10 tracks in the catalog closest to your taste profile.")

    feat_mean = df[FEATURES].mean()
    feat_std = df[FEATURES].std().replace(0, 1)
    X_scaled = (df[FEATURES] - feat_mean) / feat_std
    centroid_scaled = (centroid - feat_mean) / feat_std
    distances = np.linalg.norm(X_scaled.values - centroid_scaled.values, axis=1)
    df_rec = df.copy()
    df_rec["distance"] = distances
    df_rec["match"] = (1 / (1 + df_rec["distance"])) * 100
    recs = df_rec[~mask].sort_values("distance").head(10)

    for _, row in recs.iterrows():
        c1, c2, c3 = st.columns([4, 2, 2])
        with c1:
            st.markdown(f"**{row['track_name']}**")
            st.caption(f"{row['artists']} · {row['track_genre'].title()}")
        with c2:
            st.progress(min(row["match"] / 100, 1.0), text=f"Match: {row['match']:.0f}%")
        with c3:
            st.caption(f"Popularity: {row['popularity']}/100")

    st.markdown('</div>', unsafe_allow_html=True)

    # ---- Genre breakdown of favorites ----
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📈 Insights from your favorites")
    c1, c2 = st.columns(2)
    with c1:
        genre_counts = fav_df["track_genre"].value_counts().reset_index()
        genre_counts.columns = ["Genre", "Count"]
        fig2 = px.pie(genre_counts, names="Genre", values="Count", hole=0.5,
                       color_discrete_sequence=px.colors.sequential.Greens_r)
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#f5f5f5",
            margin=dict(l=10, r=10, t=30, b=10),
            height=320,
            title="Genres in your favorites"
        )
        st.plotly_chart(fig2, use_container_width=True)
    with c2:
        avg_compare = pd.DataFrame({
            "Feature": DISPLAY_FEATURES,
            "Your taste": [centroid[f] for f in DISPLAY_FEATURES],
            "Dataset average": [df[f].mean() for f in DISPLAY_FEATURES],
        })
        fig3 = px.bar(avg_compare, x="Feature", y=["Your taste", "Dataset average"],
                       barmode="group", color_discrete_sequence=["#1DB954", "#555555"])
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#f5f5f5",
            margin=dict(l=10, r=10, t=30, b=10),
            height=320,
            legend=dict(orientation="h", y=1.15),
            title="You vs. average listener"
        )
        st.plotly_chart(fig3, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.info("👆 Add at least 3 favorite tracks above to unlock your taste profile, fit predictions, and recommendations.")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("""
<div style="text-align:center; color:#666; padding:1.5rem 0; font-size:0.85rem;">
    Built with Streamlit · Random Forest Classifier · Spotify Tracks Dataset (Kaggle)
</div>
""", unsafe_allow_html=True)
