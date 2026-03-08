import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.load import load_processed_data

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🎵 Music Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Spotify-style dark theme
st.markdown("""
<style>
    .stApp { background-color: #191414; color: #FFFFFF; }
    .stSidebar { background-color: #121212; }
    h1, h2, h3 { color: #1DB954; }
    .metric-card {
        background: #282828;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

FEATURE_COLUMNS = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness",
    "valence", "tempo"
]

# ─────────────────────────────────────────────
# CACHE DATA — loads once, reuses on every interaction
# CONCEPT: @st.cache_data is crucial for performance.
# Without it, the app reloads 81k rows every time
# a user clicks anything. With it, data loads once
# and stays in memory. This is a real production pattern.
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = load_processed_data()
    # Load clustered version if available
    if os.path.exists("data/clustered.csv"):
        df = pd.read_csv("data/clustered.csv")
    return df

@st.cache_data
def build_similarity_matrix(df):
    """
    Pre-compute the scaled feature matrix once.
    Cosine similarity is computed on-demand per song search.
    """
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[FEATURE_COLUMNS])
    return scaled

def get_recommendations(song_name, df, feature_matrix, n=10):
    matches = df[df['track_name'].str.lower().str.contains(
        song_name.lower(), na=False)]
    if matches.empty:
        return None, None
    idx = matches.index[0]
    song_vector = feature_matrix[idx].reshape(1, -1)
    similarities = cosine_similarity(song_vector, feature_matrix)[0]
    similar_indices = similarities.argsort()[::-1][1:n+1]
    results = df.iloc[similar_indices][
        ['track_name', 'artists', 'track_genre', 'popularity']
    ].copy()
    results['similarity_score'] = (similarities[similar_indices] * 100).round(1)
    results.columns = ['Song', 'Artist', 'Genre', 'Popularity', 'Match %']
    return results, df.iloc[idx]

# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
st.sidebar.title("🎵 Music Recommender")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🎯 Song Recommender", "📊 Stats Explorer", "🔮 Cluster Explorer"]
)

df = load_data()
feature_matrix = build_similarity_matrix(df)

st.sidebar.markdown("---")
st.sidebar.metric("Total Songs", f"{len(df):,}")
st.sidebar.metric("Total Genres", df['track_genre'].nunique())
st.sidebar.metric("Total Artists", df['artists'].nunique())

# ─────────────────────────────────────────────
# PAGE 1: SONG RECOMMENDER
# ─────────────────────────────────────────────
if page == "🎯 Song Recommender":
    st.title("🎯 Song Recommender")
    st.markdown("Type a song name and get 10 similar songs based on audio features")

    col1, col2 = st.columns([3, 1])
    with col1:
        song_input = st.text_input(
            "🔍 Enter a song name",
            placeholder="e.g. Blinding Lights, Shape of You, Bohemian Rhapsody"
        )
    with col2:
        n_recs = st.slider("Number of recommendations", 5, 20, 10)

    if song_input:
        results, source_song = get_recommendations(
            song_input, df, feature_matrix, n_recs)

        if results is None:
            st.error(f"❌ Song '{song_input}' not found. Try a different name!")
        else:
            # Show source song info
            st.success(f"✅ Found: **{source_song['track_name']}** "
                      f"by {source_song['artists']} "
                      f"| Genre: {source_song['track_genre']}")

            st.markdown("### 🎵 Recommended Songs")
            st.dataframe(results, use_container_width=True, hide_index=True)

            # Bar chart of similarity scores
            fig = px.bar(
                results, x='Match %', y='Song',
                orientation='h',
                color='Match %',
                color_continuous_scale='Greens',
                title="Similarity Scores"
            )
            fig.update_layout(
                plot_bgcolor='#282828',
                paper_bgcolor='#191414',
                font_color='white',
                yaxis={'categoryorder': 'total ascending'}
            )
            st.plotly_chart(fig, use_container_width=True)

            # Radar chart comparing source song vs top recommendation
            st.markdown("### 🕸️ Audio Feature Comparison (Radar Chart)")
            top_rec_idx = df[df['track_name'] == results.iloc[0]['Song']].index
            if len(top_rec_idx) > 0:
                source_features = df.loc[source_song.name, FEATURE_COLUMNS].values
                rec_features = df.loc[top_rec_idx[0], FEATURE_COLUMNS].values

                # Normalize for radar
                scaler = MinMaxScaler()
                both = scaler.fit_transform(
                    [source_features, rec_features])

                fig2 = go.Figure()
                fig2.add_trace(go.Scatterpolar(
                    r=both[0], theta=FEATURE_COLUMNS,
                    fill='toself', name=source_song['track_name'][:30],
                    line_color='#1DB954'
                ))
                fig2.add_trace(go.Scatterpolar(
                    r=both[1], theta=FEATURE_COLUMNS,
                    fill='toself', name=results.iloc[0]['Song'][:30],
                    line_color='#FF6B6B', opacity=0.7
                ))
                fig2.update_layout(
                    polar=dict(bgcolor='#282828'),
                    paper_bgcolor='#191414',
                    font_color='white',
                    title="Audio DNA Comparison"
                )
                st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE 2: STATS EXPLORER
# ─────────────────────────────────────────────
elif page == "📊 Stats Explorer":
    st.title("📊 Stats Explorer")
    st.markdown("Explore the audio feature statistics across all 81k songs")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Danceability", f"{df['danceability'].mean():.2f}")
    col2.metric("Avg Energy", f"{df['energy'].mean():.2f}")
    col3.metric("Avg Valence", f"{df['valence'].mean():.2f}")
    col4.metric("Avg Tempo (BPM)", f"{df['tempo'].mean():.0f}")

    st.markdown("---")

    # Feature distribution
    st.markdown("### 📈 Feature Distribution")
    feature = st.selectbox("Select a feature", FEATURE_COLUMNS)
    fig = px.histogram(
        df, x=feature, nbins=50,
        color_discrete_sequence=['#1DB954'],
        title=f"Distribution of {feature.capitalize()}"
    )
    fig.update_layout(
        plot_bgcolor='#282828', paper_bgcolor='#191414', font_color='white')
    st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    st.markdown("### 🔥 Feature Correlation Heatmap")
    corr = df[FEATURE_COLUMNS].corr().round(2)
    fig2 = px.imshow(
        corr, text_auto=True,
        color_continuous_scale='RdYlGn',
        title="Audio Feature Correlations"
    )
    fig2.update_layout(paper_bgcolor='#191414', font_color='white')
    st.plotly_chart(fig2, use_container_width=True)

    # Genre analysis
    st.markdown("### 🎸 Genre Analysis")
    top_n_genres = st.slider("Number of genres to show", 5, 20, 10)
    top_genres = df['track_genre'].value_counts().head(top_n_genres)
    fig3 = px.bar(
        x=top_genres.values, y=top_genres.index,
        orientation='h',
        color=top_genres.values,
        color_continuous_scale='Greens',
        title=f"Top {top_n_genres} Genres by Song Count"
    )
    fig3.update_layout(
        plot_bgcolor='#282828', paper_bgcolor='#191414', font_color='white')
    st.plotly_chart(fig3, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE 3: CLUSTER EXPLORER
# ─────────────────────────────────────────────
elif page == "🔮 Cluster Explorer":
    st.title("🔮 Cluster Explorer")
    st.markdown("Explore how songs are grouped by audio similarity")

    if 'cluster' not in df.columns:
        st.warning("⚠️ Run clustering.py first to see this page!")
    else:
        # Cluster overview
        cluster_counts = df['cluster'].value_counts().sort_index()
        fig = px.bar(
            x=cluster_counts.index, y=cluster_counts.values,
            labels={'x': 'Cluster', 'y': 'Number of Songs'},
            color=cluster_counts.values,
            color_continuous_scale='Greens',
            title="Songs per Cluster"
        )
        fig.update_layout(
            plot_bgcolor='#282828', paper_bgcolor='#191414', font_color='white')
        st.plotly_chart(fig, use_container_width=True)

        # Explore a specific cluster
        st.markdown("### 🔍 Explore a Cluster")
        selected_cluster = st.selectbox(
            "Select cluster", sorted(df['cluster'].unique()))
        cluster_df = df[df['cluster'] == selected_cluster]

        col1, col2, col3 = st.columns(3)
        col1.metric("Songs in Cluster", len(cluster_df))
        col2.metric("Top Genre",
                   cluster_df['track_genre'].value_counts().index[0])
        col3.metric("Avg Popularity",
                   f"{cluster_df['popularity'].mean():.1f}")

        # Average audio features for this cluster
        avg_features = cluster_df[FEATURE_COLUMNS].mean()
        fig2 = go.Figure(go.Scatterpolar(
            r=avg_features.values,
            theta=FEATURE_COLUMNS,
            fill='toself',
            line_color='#1DB954'
        ))
        fig2.update_layout(
            polar=dict(bgcolor='#282828'),
            paper_bgcolor='#191414',
            font_color='white',
            title=f"Cluster {selected_cluster} — Average Audio Profile"
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Sample songs from this cluster
        st.markdown("### 🎵 Sample Songs from this Cluster")
        st.dataframe(
            cluster_df[['track_name', 'artists',
                        'track_genre', 'popularity']]
            .sort_values('popularity', ascending=False)
            .head(20),
            use_container_width=True,
            hide_index=True
        )