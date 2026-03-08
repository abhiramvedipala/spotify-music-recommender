import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.load import load_processed_data

FEATURE_COLUMNS = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness",
    "valence", "tempo"
]

def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    PURPOSE: Scale all audio features to the same 0-1 range.
    CONCEPT: MinMaxScaler normalizes features so loudness (range: -60 to 0 dB)
    doesn't dominate over danceability (range: 0 to 1).
    Without scaling, features with larger ranges unfairly influence similarity.
    This is called Feature Normalization — essential before any distance-based ML.
    """
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[FEATURE_COLUMNS])
    print(f"✅ Feature matrix built: {scaled.shape}")
    return scaled

def get_recommendations(song_name: str, df: pd.DataFrame,
                         feature_matrix: np.ndarray,
                         n: int = 10) -> pd.DataFrame:
    """
    PURPOSE: Find the top N most similar songs to a given song.
    CONCEPT: Cosine Similarity computes similarity between all song pairs.
    Steps:
    1. Find the index of the input song in our dataset
    2. Get its feature vector (row in the matrix)
    3. Compute cosine similarity between it and ALL other songs
    4. Sort by similarity score (highest = most similar)
    5. Return top N results (excluding the song itself)

    Time complexity: O(n) — we compute similarity against all 81k songs.
    In production, this would use Approximate Nearest Neighbors (ANN)
    for speed, but cosine similarity is perfect for a portfolio project.
    """
    # Find the song — case insensitive search
    matches = df[df['track_name'].str.lower() == song_name.lower()]

    if matches.empty:
        # Try partial match if exact match fails
        matches = df[df['track_name'].str.lower().str.contains(song_name.lower())]

    if matches.empty:
        print(f"❌ Song '{song_name}' not found in dataset.")
        return pd.DataFrame()

    # Take the first match
    idx = matches.index[0]
    song_vector = feature_matrix[idx].reshape(1, -1)

    # Compute cosine similarity between this song and ALL songs
    similarities = cosine_similarity(song_vector, feature_matrix)[0]

    # Get indices of top N+1 most similar (excluding itself at index 0)
    similar_indices = similarities.argsort()[::-1][1:n+1]

    # Build results DataFrame
    results = df.iloc[similar_indices][
        ['track_name', 'artists', 'track_genre', 'popularity']
    ].copy()
    results['similarity_score'] = similarities[similar_indices].round(4)
    results = results.reset_index(drop=True)

    print(f"\n🎵 Top {n} songs similar to '{df.iloc[idx]['track_name']}'")
    print(f"   Artist: {df.iloc[idx]['artists']} | Genre: {df.iloc[idx]['track_genre']}")
    print("-" * 60)
    print(results.to_string(index=False))
    return results

if __name__ == "__main__":
    df = load_processed_data()
    feature_matrix = build_feature_matrix(df)

    # Test with a few songs
    get_recommendations("Blinding Lights", df, feature_matrix)
    get_recommendations("Bohemian Rhapsody", df, feature_matrix)