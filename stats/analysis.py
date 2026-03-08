import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# This lets us import from the pipeline folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.load import load_processed_data

# Create output folder for saving charts
os.makedirs("stats/charts", exist_ok=True)

FEATURE_COLUMNS = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness",
    "valence", "tempo"
]

def summary_statistics(df: pd.DataFrame):
    """
    PURPOSE: Get mean, median, std dev, min, max for all audio features.
    CONCEPT: Descriptive stats give you a quick picture of your data
    before building any ML model. Always do this first in real jobs.
    """
    print("\n📊 SUMMARY STATISTICS FOR AUDIO FEATURES")
    print("=" * 60)
    stats = df[FEATURE_COLUMNS].describe().round(3)
    print(stats)
    return stats

def plot_feature_distributions(df: pd.DataFrame):
    """
    PURPOSE: Plot histogram for each audio feature.
    CONCEPT: Histograms show how values are distributed.
    Ex: Is danceability normally distributed or skewed?
    This matters when choosing ML algorithms.
    """
    print("\n📈 Generating feature distribution plots...")
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle("Spotify Audio Feature Distributions", fontsize=16, fontweight='bold')
    axes = axes.flatten()

    for i, feature in enumerate(FEATURE_COLUMNS):
        axes[i].hist(df[feature], bins=50, color='#1DB954', edgecolor='black', alpha=0.7)
        axes[i].set_title(feature.capitalize())
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig("stats/charts/feature_distributions.png", dpi=150)
    print("✅ Saved: stats/charts/feature_distributions.png")
    plt.close()

def plot_correlation_heatmap(df: pd.DataFrame):
    """
    PURPOSE: Show how audio features relate to each other.
    CONCEPT: Correlation tells us if two features move together.
    Ex: High energy songs tend to have high loudness (positive correlation).
    This is crucial for recommender systems — correlated features
    can be redundant, which affects our ML model quality.
    """
    print("\n🔥 Generating correlation heatmap...")
    corr_matrix = df[FEATURE_COLUMNS].corr().round(2)

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='RdYlGn',
        center=0,
        fmt='.2f',
        square=True,
        linewidths=0.5
    )
    plt.title("Audio Feature Correlation Heatmap", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("stats/charts/correlation_heatmap.png", dpi=150)
    print("✅ Saved: stats/charts/correlation_heatmap.png")
    plt.close()

def plot_popularity_vs_features(df: pd.DataFrame):
    """
    PURPOSE: See which audio features influence song popularity.
    CONCEPT: Scatter plots reveal relationships between variables.
    Ex: Do more energetic songs get more popular?
    This is descriptive analysis that drives business decisions —
    exactly what data scientists present to stakeholders.
    """
    print("\n⭐ Generating popularity vs features plots...")
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle("Popularity vs Audio Features", fontsize=16, fontweight='bold')
    axes = axes.flatten()

    # Sample 5000 rows so the plot isn't too slow
    sample = df.sample(5000, random_state=42)

    for i, feature in enumerate(FEATURE_COLUMNS):
        axes[i].scatter(sample[feature], sample['popularity'],
                       alpha=0.3, color='#1DB954', s=5)
        axes[i].set_xlabel(feature.capitalize())
        axes[i].set_ylabel("Popularity")

        # Add trend line
        z = np.polyfit(sample[feature], sample['popularity'], 1)
        p = np.poly1d(z)
        axes[i].plot(sorted(sample[feature]),
                    p(sorted(sample[feature])), "r--", alpha=0.8)

    plt.tight_layout()
    plt.savefig("stats/charts/popularity_vs_features.png", dpi=150)
    print("✅ Saved: stats/charts/popularity_vs_features.png")
    plt.close()

def genre_analysis(df: pd.DataFrame):
    """
    PURPOSE: Compare average audio features across music genres.
    CONCEPT: Group-by analysis is one of the most common operations
    in data science. It answers "how does X differ by category?"
    This is the kind of insight you'd present to a product team.
    """
    print("\n🎵 Top 10 genres by song count:")
    top_genres = df['track_genre'].value_counts().head(10)
    print(top_genres)

    # Average energy and danceability per genre (top 15 genres)
    top15 = df['track_genre'].value_counts().head(15).index
    genre_df = df[df['track_genre'].isin(top15)]
    genre_stats = genre_df.groupby('track_genre')[['energy', 'danceability', 'valence']].mean()

    genre_stats.plot(kind='bar', figsize=(14, 6), color=['#1DB954', '#191414', '#FF6B6B'])
    plt.title("Avg Energy, Danceability & Valence by Genre (Top 15)", fontweight='bold')
    plt.xlabel("Genre")
    plt.ylabel("Average Value")
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig("stats/charts/genre_analysis.png", dpi=150)
    print("✅ Saved: stats/charts/genre_analysis.png")
    plt.close()

if __name__ == "__main__":
    df = load_processed_data()
    summary_statistics(df)
    plot_feature_distributions(df)
    plot_correlation_heatmap(df)
    plot_popularity_vs_features(df)
    genre_analysis(df)
    print("\n🎉 All stats generated! Check stats/charts/ folder.")