import numpy as np
import pandas as pd
import pytest

from src.clustering import (
    cluster_panels,
    cluster_panels_auto,
    compute_cluster_mean_ts,
    compute_umap_embedding,
    extract_panel_features,
)


@pytest.fixture()
def multi_panel_df() -> pd.DataFrame:
    """5 панелей по 24 точки с разными паттернами."""
    rng = np.random.default_rng(42)
    rows = []
    dates = pd.date_range("2022-01-01", periods=24, freq="MS")
    for i in range(5):
        base = (i + 1) * 10.0
        values = base + rng.normal(0, 1, size=24)
        for d, v in zip(dates, values):
            rows.append({"article": f"A{i}", "date": d, "sales": max(0.0, v)})
    return pd.DataFrame(rows)


@pytest.fixture()
def features_df(multi_panel_df) -> pd.DataFrame:
    return extract_panel_features(multi_panel_df, "article", "sales")


class TestExtractPanelFeatures:
    def test_returns_dataframe(self, multi_panel_df):
        result = extract_panel_features(multi_panel_df, "article", "sales")
        assert isinstance(result, pd.DataFrame)

    def test_index_is_panel_ids(self, multi_panel_df):
        result = extract_panel_features(multi_panel_df, "article", "sales")
        assert set(result.index) == {"A0", "A1", "A2", "A3", "A4"}

    def test_autocorr_lag12_column_present(self, multi_panel_df):
        result = extract_panel_features(multi_panel_df, "article", "sales")
        assert "autocorr_lag12" in result.columns

    def test_normalized_zero_mean(self, multi_panel_df):
        """StandardScaler → среднее по столбцу ≈ 0."""
        result = extract_panel_features(multi_panel_df, "article", "sales")
        col_means = result.mean()
        assert (col_means.abs() < 1e-10).all()

    def test_no_nan(self, multi_panel_df):
        result = extract_panel_features(multi_panel_df, "article", "sales")
        assert not result.isna().any().any()


class TestClusterPanels:
    def test_returns_series(self, features_df):
        result = cluster_panels(features_df, n_clusters=3, method="kmeans")
        assert isinstance(result, pd.Series)

    def test_index_matches_features(self, features_df):
        result = cluster_panels(features_df, n_clusters=3, method="kmeans")
        assert set(result.index) == set(features_df.index)

    def test_n_clusters_respected(self, features_df):
        result = cluster_panels(features_df, n_clusters=2, method="kmeans")
        assert result.nunique() == 2

    def test_labels_are_integers(self, features_df):
        result = cluster_panels(features_df, n_clusters=3, method="kmeans")
        assert result.dtype in (np.int32, np.int64, int)

    def test_invalid_method_raises(self, features_df):
        with pytest.raises(Exception):
            cluster_panels(features_df, n_clusters=3, method="unknown_algo")


class TestComputeUmapEmbedding:
    def test_shape(self, features_df):
        embedding = compute_umap_embedding(features_df, random_state=0)
        assert embedding.shape == (len(features_df), 2)

    def test_no_nan(self, features_df):
        embedding = compute_umap_embedding(features_df, random_state=0)
        assert np.all(np.isfinite(embedding))


class TestComputeClusterMeanTs:
    def test_returns_dataframe(self, multi_panel_df, features_df):
        labels = cluster_panels(features_df, n_clusters=2, method="kmeans")
        result = compute_cluster_mean_ts(multi_panel_df, "article", "date", "sales", labels)
        assert isinstance(result, pd.DataFrame)

    def test_columns(self, multi_panel_df, features_df):
        labels = cluster_panels(features_df, n_clusters=2, method="kmeans")
        result = compute_cluster_mean_ts(multi_panel_df, "article", "date", "sales", labels)
        assert set(result.columns) == {"cluster_id", "date", "mean_value"}

    def test_cluster_ids_are_int(self, multi_panel_df, features_df):
        labels = cluster_panels(features_df, n_clusters=2, method="kmeans")
        result = compute_cluster_mean_ts(multi_panel_df, "article", "date", "sales", labels)
        assert result["cluster_id"].dtype in (np.int32, np.int64, int)

    def test_n_dates_per_cluster(self, multi_panel_df, features_df):
        """Каждый кластер должен иметь 24 даты (все временные точки)."""
        labels = cluster_panels(features_df, n_clusters=2, method="kmeans")
        result = compute_cluster_mean_ts(multi_panel_df, "article", "date", "sales", labels)
        for _, grp in result.groupby("cluster_id"):
            assert len(grp) == 24

    def test_mean_values_finite(self, multi_panel_df, features_df):
        labels = cluster_panels(features_df, n_clusters=2, method="kmeans")
        result = compute_cluster_mean_ts(multi_panel_df, "article", "date", "sales", labels)
        assert np.all(np.isfinite(result["mean_value"].values))


class TestClusterPanelsAuto:
    def test_returns_tuple(self, features_df):
        labels, scores, best_k = cluster_panels_auto(features_df, max_k=4)
        assert isinstance(labels, pd.Series)
        assert isinstance(scores, dict)
        assert isinstance(best_k, int)

    def test_best_k_in_range(self, features_df):
        _, scores, best_k = cluster_panels_auto(features_df, max_k=4)
        assert 2 <= best_k <= 4

    def test_scores_keys_cover_range(self, features_df):
        _, scores, _ = cluster_panels_auto(features_df, max_k=4)
        assert set(scores.keys()) == {2, 3, 4}

    def test_scores_bounded(self, features_df):
        _, scores, _ = cluster_panels_auto(features_df, max_k=4)
        for s in scores.values():
            assert -1.0 <= s <= 1.0

    def test_labels_nunique_equals_best_k(self, features_df):
        labels, _, best_k = cluster_panels_auto(features_df, max_k=4)
        assert labels.nunique() == best_k

    def test_labels_index_matches(self, features_df):
        labels, _, _ = cluster_panels_auto(features_df, max_k=4)
        assert set(labels.index) == set(features_df.index)
