import plotly.graph_objects as go
import seaborn as sns
from matplotlib import pyplot as plt

from src.evaluation import EvaluationResults, get_panel_metrics_wide


def plot_panel_metrics_distributions(
    results: EvaluationResults,
    metrics_to_plot: list[str] = ["mape", "rmse", "mae", "r2"],
    figsize: tuple[int, int] = (18, 5),
) -> None:
    """Отображает распределения метрик по панелям для каждого сплита."""
    panel_metrics_df = results.get_panel_metrics_df()
    panel_metrics_wide = get_panel_metrics_wide(panel_metrics_df, sort_metric="mape")
    split_names = list(panel_metrics_wide.keys())

    for metric in metrics_to_plot:
        fig, axes = plt.subplots(1, len(split_names), figsize=figsize)

        if len(split_names) == 1:
            axes = [axes]

        for idx, split_name in enumerate(split_names):
            ax = axes[idx]

            metric_data = panel_metrics_wide[split_name][metric]

            sns.histplot(data=metric_data, bins=30, kde=True, ax=ax)

            median_val = metric_data.median()
            mean_val = metric_data.mean()

            ax.axvline(
                median_val,
                color="red",
                linestyle="--",
                label=f"Медиана: {median_val:.4f}",
            )
            ax.axvline(
                mean_val,
                color="green",
                linestyle="--",
                label=f"Среднее: {mean_val:.4f}",
            )
            ax.set_title(f"{split_name.upper()} - {metric.upper()}")
            ax.set_xlabel(metric.upper())
            ax.set_ylabel("Частота")
            ax.legend()

        plt.tight_layout()
        plt.show()


def plot_worst_panels(
    results: EvaluationResults,
    metric: str = "mape",
    n_worst: int = 10,
    split_name: str = "test",
) -> None:
    """Отображает худшие панели по выбранной метрике."""
    panel_metrics_df = results.get_panel_metrics_df()
    split_data = panel_metrics_df[panel_metrics_df["split"] == split_name].copy()
    worst = split_data.nlargest(n_worst, metric)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(worst["panel_id"].astype(str), worst[metric])
    ax.set_xlabel(metric.upper())
    ax.set_ylabel("Panel ID")
    ax.set_title(f"Top {n_worst} Worst Panels by {metric.upper()} ({split_name})")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()


def plot_best_panels(
    results: EvaluationResults,
    metric: str = "mape",
    n_best: int = 10,
    split_name: str = "test",
) -> None:
    """Отображает лучшие панели по выбранной метрике."""
    panel_metrics_df = results.get_panel_metrics_df()
    split_data = panel_metrics_df[panel_metrics_df["split"] == split_name].copy()
    best = split_data.nsmallest(n_best, metric)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(best["panel_id"].astype(str), best[metric])
    ax.set_xlabel(metric.upper())
    ax.set_ylabel("Panel ID")
    ax.set_title(f"Top {n_best} Best Panels by {metric.upper()} ({split_name})")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()


def plot_overall_metrics_comparison(
    results: EvaluationResults,
    metrics_to_plot: list[str] = ["mape", "rmse", "mae", "r2"],
    figsize: tuple[int, int] = (14, 6),
) -> None:
    """Отображает сравнение общих метрик между сплитами."""
    overall_df = results.get_overall_metrics_df()

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=figsize)

    if len(metrics_to_plot) == 1:
        axes = [axes]

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        ax.bar(overall_df["split"], overall_df[metric])
        ax.set_xlabel("Split")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} by Split")
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()


def plot_panel_predictions(
    panel_id: int | str,
    results: EvaluationResults,
    interactive: bool = True,
    figsize: tuple[int, int] = (14, 5),
) -> go.Figure | None:
    """Отображает предсказания для одной панели."""
    if interactive:
        return _plot_panel_predictions_plotly(panel_id, results)
    else:
        _plot_panel_predictions_matplotlib(panel_id, results, figsize)
        return None


def _plot_panel_predictions_plotly(
    panel_id: int | str,
    results: EvaluationResults,
) -> go.Figure:
    """Создает интерактивный график сравнения предсказаний с Plotly."""
    fig = go.Figure()
    colors = ["blue", "orange", "green", "red", "purple"]

    start = 0
    overall_mape = []
    prev_end_true = None
    prev_end_pred = None
    prev_end_idx = None

    for i, split_eval in enumerate(results.splits):
        panel_metric = None
        for pm in split_eval.panel_metrics:
            if pm.panel_id == panel_id:
                panel_metric = pm
                break

        if panel_metric is None:
            continue

        y_true = panel_metric.y_true
        y_pred = panel_metric.y_pred
        count = len(y_true)
        end = start + count

        color = colors[i % len(colors)]
        split_name = split_eval.split_name.title()

        if prev_end_true is not None:
            fig.add_trace(
                go.Scatter(
                    x=[prev_end_idx, start],
                    y=[prev_end_true, y_true[0]],
                    mode="lines",
                    line=dict(color="gray", dash="dot"),
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[prev_end_idx, start],
                    y=[prev_end_pred, y_pred[0]],
                    mode="lines",
                    line=dict(color="gray", dash="dot"),
                    showlegend=False,
                )
            )

        fig.add_trace(
            go.Scatter(
                x=list(range(start, end)),
                y=y_true,
                mode="lines+markers",
                name=f"{split_name} True",
                line=dict(color=color),
                marker=dict(size=6),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=list(range(start, end)),
                y=y_pred,
                mode="lines+markers",
                name=f"{split_name} Pred",
                line=dict(color=color, dash="dash"),
                marker=dict(size=4, symbol="x"),
            )
        )

        overall_mape.append(f"{split_name}: {panel_metric.metrics.mape:.4f}")
        prev_end_true = y_true[-1]
        prev_end_pred = y_pred[-1]
        prev_end_idx = end - 1
        start = end

    mape_text = " | ".join(overall_mape)
    fig.update_layout(
        title=f"Predictions vs True Values for Panel {panel_id}<br><sub>MAPE: {mape_text}</sub>",
        xaxis_title="Time Steps",
        yaxis_title="Value",
        width=1200,
        height=600,
        hovermode="x unified",
    )

    return fig


def _plot_panel_predictions_matplotlib(
    panel_id: int | str,
    results: EvaluationResults,
    figsize: tuple[int, int],
) -> None:
    """Отображает предсказания для одной панели с Matplotlib."""
    fig, axes = plt.subplots(1, len(results.splits), figsize=figsize, sharey=True)

    if len(results.splits) == 1:
        axes = [axes]

    for idx, split_eval in enumerate(results.splits):
        ax = axes[idx]

        panel_metric = None
        for pm in split_eval.panel_metrics:
            if pm.panel_id == panel_id:
                panel_metric = pm
                break

        if panel_metric is None:
            ax.set_title(f"{split_eval.split_name.upper()} - Panel not found")
            continue

        y_true = panel_metric.y_true
        y_pred = panel_metric.y_pred

        ax.plot(y_true, label="True", marker="o", alpha=0.7)
        ax.plot(y_pred, label="Predicted", marker="x", alpha=0.7)
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.set_title(
            f"{split_eval.split_name.upper()} - Panel {panel_id}\n"
            f"MAPE: {panel_metric.metrics.mape:.4f}"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_best_predictions(
    results: EvaluationResults,
    n_best: int = 5,
    metric: str = "mape",
    split_name: str = "test",
    interactive: bool = True,
) -> None:
    """Отображает предсказания для лучших панелей."""
    panel_metrics_df = results.get_panel_metrics_df()
    panel_metrics_wide = get_panel_metrics_wide(panel_metrics_df, sort_metric=metric)

    best_panels = panel_metrics_wide[split_name].nsmallest(n_best, metric)

    for _, row in best_panels.iterrows():
        panel_id = row["panel_id"]
        fig = plot_panel_predictions(panel_id, results, interactive=interactive)
        if interactive and fig is not None:
            fig.show()


def plot_worst_predictions(
    results: EvaluationResults,
    n_worst: int = 5,
    metric: str = "mape",
    split_name: str = "test",
    interactive: bool = True,
) -> None:
    """Отображает предсказания для худших панелей."""
    panel_metrics_df = results.get_panel_metrics_df()
    panel_metrics_wide = get_panel_metrics_wide(panel_metrics_df, sort_metric=metric)

    worst_panels = panel_metrics_wide[split_name].nlargest(n_worst, metric)

    for _, row in worst_panels.iterrows():
        panel_id = row["panel_id"]
        fig = plot_panel_predictions(panel_id, results, interactive=interactive)
        if interactive and fig is not None:
            fig.show()
