import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.load import load_processed_data

FEATURE_COLUMNS = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness",
    "valence", "tempo"
]

def scale_features(df: pd.DataFrame) -> np.ndarray:
    """
    PURPOSE: Normalize features before clustering.
    Same reason as content_based.py — KMeans uses
    Euclidean distance, so unscaled features cause bias.
    """
    scaler = MinMaxScaler()
    return scaler.fit_transform(df[FEATURE_COLUMNS])

def find_optimal_k(scaled_matrix: np.ndarray, max_k: int = 15):
    """
    PURPOSE: Find the best number of clusters using the Elbow Method.
    CONCEPT: We run KMeans with k=2 to k=15 and plot the inertia
    (sum of squared distances from each point to its cluster center).
    The 'elbow' in the curve = the optimal k where adding more
    clusters gives diminishing returns. This is a standard technique
    every ML engineer should know.
    """
    print("🔍 Finding optimal number of clusters (Elbow Method)...")
    inertias = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(scaled_matrix)
        inertias.append(km.inertia_)
        print(f"   k={k}: inertia={km.inertia_:.0f}")

    # Plot elbow curve
    plt.figure(figsize=(10, 5))
    plt.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method — Optimal Number of Clusters")
    plt.xticks(k_range)
    plt.grid(True, alpha=0.3)
    os.makedirs("stats/charts", exist_ok=True)
    plt.savefig("stats/charts/elbow_curve.png", dpi=150)
    print("✅ Saved: stats/charts/elbow_curve.png")
    plt.close()

def train_kmeans(scaled_matrix: np.ndarray, k: int = 10) -> KMeans:
    """
    PURPOSE: Train the KMeans model with k clusters.
    CONCEPT: KMeans assigns each of the 81k songs to one of k clusters.
    Songs in the same cluster have similar audio features.
    n_init=10 means we run KMeans 10 times with different starting points
    and pick the best result — this avoids bad random initializations.
    """
    print(f"\n🎯 Training KMeans with k={k} clusters...")
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(scaled_matrix)
    print(f"✅ KMeans trained! Inertia: {km.inertia_:.0f}")
    return km

def visualize_clusters(scaled_matrix: np.ndarray, labels: np.ndarray):
    """
    PURPOSE: Visualize clusters in 2D using PCA.
    CONCEPT: Our feature matrix has 9 dimensions (9 audio features).
    We can't plot 9D data directly. PCA (Principal Component Analysis)
    reduces dimensions to 2 while preserving as much variance as possible.
    This lets us SEE the clusters — great for presentations and portfolios.
    """
    print("\n🎨 Generating cluster visualization...")
    pca = PCA(n_components=2, random_state=42)
    # Sample 5000 points so plot isn't too slow
    sample_idx = np.random.choice(len(scaled_matrix), 5000, replace=False)
    reduced = pca.fit_transform(scaled_matrix[sample_idx])

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1],
                         c=labels[sample_idx], cmap='tab10',
                         alpha=0.6, s=10)
    plt.colorbar(scatter, label='Cluster')
    plt.title("Song Clusters (PCA 2D Visualization)", fontsize=14, fontweight='bold')
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.savefig("stats/charts/clusters_pca.png", dpi=150)
    print("✅ Saved: stats/charts/clusters_pca.png")
    plt.close()

def add_cluster_labels(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """
    PURPOSE: Add the cluster ID to each song in the DataFrame.
    This lets the recommender say 'give me other songs in cluster 3'
    instead of computing similarity against all 81k songs.
    Clusters act as a fast pre-filter — a real performance optimization.
    """
    df = df.copy()
    df['cluster'] = labels
    print(f"\n📊 Songs per cluster:")
    print(df['cluster'].value_counts().sort_index())
    return df

if __name__ == "__main__":
    df = load_processed_data()
    scaled = scale_features(df)
    find_optimal_k(scaled, max_k=12)
    km = train_kmeans(scaled, k=10)
    df_clustered = add_cluster_labels(df, km.labels_)
    visualize_clusters(scaled, km.labels_)
    df_clustered.to_csv("data/clustered.csv", index=False)
    print("\n💾 Saved clustered data to data/clustered.csv")