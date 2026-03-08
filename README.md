#  Spotify Music Recommender

> A full-stack data/ML pipeline that analyzes 81,000+ songs and delivers personalized music recommendations using content-based filtering, KMeans clustering, and a live Streamlit dashboard — with AWS S3 simulation via LocalStack.

![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?style=flat-square&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![AWS](https://img.shields.io/badge/AWS%20S3-LocalStack-FF9900?style=flat-square&logo=amazon-aws&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-7%2F7%20Passing-brightgreen?style=flat-square)

---

##  Overview

This project simulates a production-grade music recommendation system — the kind used in real streaming platforms. It covers the full ML pipeline from raw data ingestion to a live interactive dashboard, demonstrating skills in data engineering, machine learning, statistical analysis, and cloud storage.

**Dataset**: [Kaggle Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) — 114k songs across 113 genres

---

##  Architecture

```
Raw CSV (114k songs)
       ↓
  ETL Pipeline          ← pandas, data cleaning, deduplication
       ↓
Processed Data (81k)
       ↓
  ┌────┴────┐
  ↓         ↓
Stats     Recommender
Analysis  Engine
  ↓         ↓
4 Charts  Cosine Similarity + KMeans Clustering
       ↓
  Streamlit Dashboard   ← live interactive UI
       ↓
  AWS S3 Data Lake      ← LocalStack simulation
```

---

##  Features

###  ETL Pipeline
- Ingests raw Kaggle CSV (114k songs)
- Cleans nulls, removes duplicates → 81,343 clean records
- Extracts 15 audio features per track

###  Descriptive Statistics
- Summary stats across 9 audio features (danceability, energy, tempo, etc.)
- Feature distribution histograms
- Correlation heatmap
- Popularity vs. audio features scatter plots
- Genre analysis (113 genres)

### Recommendation Engine
- **Content-Based Filtering**: MinMaxScaler normalization + cosine similarity
- Achieves 99.9%+ similarity scores on known songs
- **KMeans Clustering**: Elbow method to find optimal k=10
- PCA 2D visualization of song clusters

### Streamlit Dashboard (3 pages)
- **Song Recommender**: Type any song → get top 10 similar tracks instantly
- **Stats Explorer**: Interactive charts and dataset insights
- **Cluster Explorer**: Explore the 10 music clusters with audio profiles

###  AWS S3 Simulation
- LocalStack Docker container simulating real AWS S3
- Uploads raw, processed, and clustered datasets to `s3://spotify-data-lake`
- Demonstrates cloud data lake architecture

###  Unit Tests
- 7 tests covering data integrity, recommender accuracy, and edge cases
- All 7/7 passing with pytest

---

##  Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.13 |
| Data Processing | pandas, NumPy |
| Machine Learning | scikit-learn (KMeans, cosine similarity, MinMaxScaler, PCA) |
| Visualization | matplotlib, seaborn, plotly |
| Dashboard | Streamlit |
| Cloud (simulated) | AWS S3, boto3, LocalStack, Docker |
| Testing | pytest |
| Version Control | Git, GitHub |

---

##  Getting Started

### Prerequisites
- Python 3.10+
- Docker Desktop (for AWS simulation)

### Installation

```bash
# Clone the repo
git clone https://github.com/abhiramvedipala/spotify-music-recommender.git
cd spotify-music-recommender

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### Download the Dataset

```bash
# Set up Kaggle API credentials (~/.kaggle/kaggle.json)
kaggle datasets download -d maharshipandya/-spotify-tracks-dataset -p data/raw --unzip
```

### Run the Pipeline

```bash
# 1. Extract & Transform
python3 pipeline/extract.py
python3 pipeline/transform.py

# 2. Descriptive Statistics
python3 stats/analysis.py

# 3. Build Recommender + Clusters
python3 recommender/content_based.py
python3 recommender/clustering.py

# 4. Launch Dashboard
streamlit run dashboard/app.py

# 5. AWS S3 Simulation (requires Docker)
docker run -d -p 4566:4566 localstack/localstack
python3 aws_sim/s3_pipeline.py
```

### Run Tests

```bash
python3 -m pytest tests/test_pipeline.py -v
```

---

##  Project Structure

```
spotify-recommender/
├── pipeline/
│   ├── extract.py          # Load raw CSV, inspect data
│   ├── transform.py        # Clean, dedupe, select features
│   └── load.py             # Single source of truth for data loading
├── recommender/
│   ├── content_based.py    # Cosine similarity recommender
│   └── clustering.py       # KMeans + elbow method + PCA
├── stats/
│   ├── analysis.py         # Descriptive stats + 4 charts
│   └── charts/             # Generated visualizations
├── dashboard/
│   └── app.py              # Streamlit web app (3 pages)
├── aws_sim/
│   └── s3_pipeline.py      # LocalStack S3 simulation
├── tests/
│   └── test_pipeline.py    # 7 unit tests (all passing)
├── data/
│   ├── raw/                # Original Kaggle dataset
│   ├── processed.csv       # Cleaned 81k songs
│   └── clustered.csv       # With KMeans cluster labels
└── requirements.txt
```

---

##  Results

| Metric | Value |
|---|---|
| Songs processed | 81,343 |
| Genres covered | 113 |
| Unique artists | 31,437 |
| Recommendation similarity | 99.9%+ |
| Clusters identified | 10 |
| Unit tests passing | 7/7 |
| S3 data lake size | ~43 MB |

---

##  Author

**Abhiram Vedipala**  
Computer Science Student @ Florida International University  
[GitHub](https://github.com/abhiramvedipala) · [LinkedIn](https://linkedin.com/in/abhiramvedipala)
