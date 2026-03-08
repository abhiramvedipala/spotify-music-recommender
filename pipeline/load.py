import pandas as pd
import os

def load_processed_data(filepath: str = "data/processed.csv") -> pd.DataFrame:
    """
    PURPOSE: Load the cleaned processed data for use by other modules.
    CONCEPT: This is the L in ETL — Load.
    Every other module (recommender, stats, dashboard)
    imports this function to get clean data. One source of truth.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"❌ Processed data not found at {filepath}. "
            "Run transform.py first!"
        )
    df = pd.read_csv(filepath)
    print(f"✅ Loaded processed data: {df.shape[0]} songs, {df.shape[1]} columns")
    return df

if __name__ == "__main__":
    df = load_processed_data()
    print(df.head(3))