import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def plot_cluster_umap(
    embedding: np.ndarray,
    labels: np.ndarray,
    title: str = "",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Визуализирует кластеры на UMAP проекции."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap="tab20", s=10, alpha=0.7)
    plt.colorbar(scatter, ax=ax, label="cluster")
    ax.set_title(title)
    return ax


def plot_cluster_profiles_heatmap(
    df: pd.DataFrame,
    cluster_col: str,
    feature_cols: list[str],
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Визуализирует профили кластеров как heatmap."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))

    profiles = df.groupby(cluster_col)[feature_cols].mean()
    profiles_norm = (profiles - profiles.min()) / (profiles.max() - profiles.min() + 1e-9)

    sns.heatmap(profiles_norm, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax)
    ax.set_title(f"Профили кластеров ({cluster_col})")
    return ax


def plot_cluster_samples(
    df: pd.DataFrame,
    ts_df: pd.DataFrame,
    cluster_col: str,
    article_col: str = "article",
    value_col: str = "sales",
    n_samples: int = 5,
    random_state: int = 42,
) -> plt.Figure:
    """Визуализирует случайные временные ряды из каждого кластера."""
    rng = np.random.default_rng(random_state)
    clusters = sorted([c for c in df[cluster_col].unique() if c != -1])

    fig, axes = plt.subplots(len(clusters), n_samples, figsize=(15, 2.5 * len(clusters)))
    if len(clusters) == 1:
        axes = axes.reshape(1, -1)

    for i, cluster in enumerate(clusters):
        articles = df[df[cluster_col] == cluster][article_col].tolist()
        sample_size = min(n_samples, len(articles))
        sample_articles = rng.choice(articles, size=sample_size, replace=False)

        for j, article in enumerate(sample_articles):
            series = ts_df[ts_df[article_col] == article][value_col].values
            if len(series) > 0:
                axes[i, j].plot(series)
            if j == 0:
                axes[i, j].set_ylabel(f"cluster {cluster}")

        for j in range(sample_size, n_samples):
            axes[i, j].axis("off")

    fig.suptitle(f"Примеры рядов по кластерам ({cluster_col})")
    fig.tight_layout()
    return fig


def plot_clustering_report(
    features_df: pd.DataFrame,
    ts_df: pd.DataFrame,
    embedding: np.ndarray,
    cluster_col: str,
    feature_cols: list[str],
    title: str = "",
) -> None:
    """Выводит полный отчёт по кластеризации: UMAP, профили, примеры рядов."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    plot_cluster_umap(embedding, features_df[cluster_col], title=title or cluster_col, ax=axes[0])
    plot_cluster_profiles_heatmap(features_df, cluster_col, feature_cols, ax=axes[1])

    fig.tight_layout()
    plt.show()

    plot_cluster_samples(features_df, ts_df, cluster_col)
    plt.show()
