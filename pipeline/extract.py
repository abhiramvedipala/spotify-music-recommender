import pandas as pd

def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    PURPOSE: Read the raw CSV into a DataFrame.
    CONCEPT: This is the Extract step of ETL.
    We only READ here — never modify the raw data.
    """
    df = pd.read_csv(filepath)
    print(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns")
    print(f"\n📋 Columns: {list(df.columns)}")
    print(f"\n🔍 First 3 rows:\n{df.head(3)}")
    print(f"\n❓ Missing values:\n{df.isnull().sum()}")
    return df

if __name__ == "__main__":
    df = load_raw_data("data/raw/dataset.csv")