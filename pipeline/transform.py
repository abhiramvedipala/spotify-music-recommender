import pandas as pd

# These are the audio features Spotify uses to describe every song.
# We'll use these columns to build our recommendation engine.
FEATURE_COLUMNS = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness",
    "valence", "tempo"
]

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    PURPOSE: Clean and shape raw data for ML.
    CONCEPT: This is the Transform step of ETL.
    We drop nulls, remove duplicates, and keep
    only the columns relevant to our recommender.
    """
    print(f" Starting shape: {df.shape}")

    # Drop rows with missing track or album names
    df = df.dropna(subset=["track_name", "album_name"])
    print(f"After dropping nulls: {df.shape}")

    # Remove duplicate songs (same track name + same artist)
    df = df.drop_duplicates(subset=["track_name", "artists"])
    print(f"After removing duplicates: {df.shape}")

    # Reset index so row numbers are clean after dropping rows
    df = df.reset_index(drop=True)

    # Keep only columns we actually need
    keep_cols = ["track_id", "track_name", "artists",
                 "album_name", "track_genre", "popularity"] + FEATURE_COLUMNS
    df = df[keep_cols]
    print(f"Final columns: {list(df.columns)}")
    print(f"Clean dataset shape: {df.shape}")

    return df


if __name__ == "__main__":
    raw = pd.read_csv("data/raw/dataset.csv")
    clean = transform_data(raw)
    # Save the cleaned data for next steps
    clean.to_csv("data/processed.csv", index=False)
    print(f" Saved to data/processed.csv")
    