import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
from pipeline.load import load_processed_data
from recommender.content_based import get_recommendations, build_feature_matrix

@pytest.fixture(scope="module")
def data():
    return load_processed_data()

@pytest.fixture(scope="module")
def feature_matrix(data):
    return build_feature_matrix(data)

def test_data_loads(data):
    assert data is not None
    assert len(data) > 0

def test_data_has_required_columns(data):
    for col in ['track_name', 'artists', 'danceability', 'energy', 'tempo']:
        assert col in data.columns

def test_no_nulls_in_features(data):
    assert data[['danceability', 'energy', 'loudness', 'tempo']].isnull().sum().sum() == 0

def test_data_shape(data):
    assert data.shape[0] > 50000
    assert data.shape[1] >= 10

def test_recommender_returns_results(data, feature_matrix):
    results = get_recommendations("Blinding Lights", data, feature_matrix)
    assert results is not None and len(results) > 0

def test_recommender_has_correct_columns(data, feature_matrix):
    results = get_recommendations("Blinding Lights", data, feature_matrix)
    assert 'track_name' in results.columns
    assert 'artists' in results.columns

def test_recommender_unknown_song(data, feature_matrix):
    results = get_recommendations("zzz_fake_song_xyz_123", data, feature_matrix)
    assert results is None or len(results) == 0