"""Visualization and reporting for evaluation results.

Generates summary tables, bar charts, heatmaps, and threshold sweep plots
from the results DataFrame produced by :class:`EvaluationRunner`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script usage

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.evaluation.metrics import (
    attack_success_rate,
    defense_block_rate,
    false_positive_rate,
    latency_overhead,
)


# Consistent color palette across plots
_PALETTE = sns.color_palette("Set2", 8)


class EvaluationReporter:
    """Generates charts and summary tables from evaluation results.

    Parameters
    ----------
    results_df:
        The DataFrame produced by :meth:`EvaluationRunner.run_full_evaluation`.
        Must contain columns: ``config_name``, ``query``, ``query_type``,
        ``response``, ``success``, ``timing_total``.
    """

    def __init__(self, results_df: pd.DataFrame) -> None:
        self._df = results_df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def _attack_df(self) -> pd.DataFrame:
        return self._df[self._df["query_type"] != "benign"]

    @property
    def _benign_df(self) -> pd.DataFrame:
        return self._df[self._df["query_type"] == "benign"]

    @property
    def _config_names(self) -> list[str]:
        return list(self._df["config_name"].unique())

    @property
    def _attack_types(self) -> list[str]:
        return list(self._attack_df["query_type"].unique())

    @staticmethod
    def _save_fig(fig: plt.Figure, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------

    def summary_table(self) -> pd.DataFrame:
        """Pivot table of Attack Success Rate by config x attack_type.

        Returns
        -------
        pd.DataFrame
            Rows = config_name, columns = attack_type, values = ASR (0-1).
            An extra ``overall`` column gives the aggregate ASR per config.
        """
        records: list[dict[str, Any]] = []

        for config in self._config_names:
            cfg_attack = self._attack_df[
                self._attack_df["config_name"] == config
            ]
            row: dict[str, Any] = {"config_name": config}

            for atype in self._attack_types:
                subset = cfg_attack[cfg_attack["query_type"] == atype]
                asr = attack_success_rate(subset.to_dict("records"))
                row[atype] = round(asr, 4)

            # Overall ASR across all attack types
            overall_asr = attack_success_rate(cfg_attack.to_dict("records"))
            row["overall"] = round(overall_asr, 4)

            # FPR for this config
            cfg_benign = self._benign_df[
                self._benign_df["config_name"] == config
            ]
            fpr = false_positive_rate(cfg_benign.to_dict("records"))
            row["fpr"] = round(fpr, 4)

            # Average latency
            avg_latency = cfg_attack["timing_total"].mean()
            row["avg_latency_s"] = round(avg_latency, 3) if pd.notna(avg_latency) else 0.0

            records.append(row)

        return pd.DataFrame(records).set_index("config_name")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_asr_comparison(self, ax: plt.Axes | None = None) -> plt.Figure:
        """Bar chart of overall ASR across defense configurations.

        Returns the matplotlib Figure.
        """
        summary = self.summary_table()

        fig, ax_ = (ax.figure, ax) if ax is not None else plt.subplots(
            figsize=(10, 6)
        )
        configs = summary.index.tolist()
        asr_values = summary["overall"].values

        bars = ax_.bar(configs, asr_values, color=_PALETTE[: len(configs)])
        ax_.set_ylabel("Attack Success Rate")
        ax_.set_title("Attack Success Rate by Defense Configuration")
        ax_.set_ylim(0, 1.05)
        ax_.set_xticklabels(configs, rotation=30, ha="right")

        # Add value labels on bars
        for bar, val in zip(bars, asr_values):
            ax_.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.1%}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        fig.tight_layout()
        return fig

    def plot_defense_heatmap(self, ax: plt.Axes | None = None) -> plt.Figure:
        """Heatmap of defense block rate by defense config x attack_type.

        Block rate is computed relative to the 'none' configuration.

        Returns the matplotlib Figure.
        """
        configs = self._config_names
        attack_types = self._attack_types

        # Get baseline (no-defense) results
        none_attack = self._attack_df[self._attack_df["config_name"] == "none"]
        none_results_by_type: dict[str, list[dict]] = {}
        for atype in attack_types:
            subset = none_attack[none_attack["query_type"] == atype]
            none_results_by_type[atype] = subset.to_dict("records")

        # Build block rate matrix
        matrix: list[list[float]] = []
        row_labels: list[str] = []

        for config in configs:
            if config == "none":
                continue
            row_labels.append(config)
            row: list[float] = []
            cfg_attack = self._attack_df[
                self._attack_df["config_name"] == config
            ]
            for atype in attack_types:
                defense_subset = cfg_attack[
                    cfg_attack["query_type"] == atype
                ].to_dict("records")
                baseline_subset = none_results_by_type.get(atype, [])
                br = defense_block_rate(baseline_subset, defense_subset)
                row.append(round(br, 4))
            matrix.append(row)

        heatmap_df = pd.DataFrame(
            matrix, index=row_labels, columns=attack_types
        )

        fig, ax_ = (ax.figure, ax) if ax is not None else plt.subplots(
            figsize=(10, 6)
        )
        sns.heatmap(
            heatmap_df,
            annot=True,
            fmt=".1%",
            cmap="YlGn",
            vmin=0,
            vmax=1,
            ax=ax_,
            linewidths=0.5,
        )
        ax_.set_title("Defense Block Rate by Configuration and Attack Type")
        ax_.set_ylabel("Defense Configuration")
        ax_.set_xlabel("Attack Type")

        fig.tight_layout()
        return fig

    def plot_fpr_comparison(self, ax: plt.Axes | None = None) -> plt.Figure:
        """Bar chart of false positive rate across defense configurations.

        Returns the matplotlib Figure.
        """
        summary = self.summary_table()

        fig, ax_ = (ax.figure, ax) if ax is not None else plt.subplots(
            figsize=(10, 6)
        )
        configs = summary.index.tolist()
        fpr_values = summary["fpr"].values

        bars = ax_.bar(configs, fpr_values, color=_PALETTE[: len(configs)])
        ax_.set_ylabel("False Positive Rate")
        ax_.set_title("False Positive Rate by Defense Configuration")
        ax_.set_ylim(0, max(0.3, max(fpr_values) * 1.3) if len(fpr_values) else 0.3)
        ax_.set_xticklabels(configs, rotation=30, ha="right")

        for bar, val in zip(bars, fpr_values):
            ax_.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.1%}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        fig.tight_layout()
        return fig

    def plot_latency_comparison(
        self, ax: plt.Axes | None = None
    ) -> plt.Figure:
        """Bar chart of average latency by defense configuration.

        Returns the matplotlib Figure.
        """
        summary = self.summary_table()

        fig, ax_ = (ax.figure, ax) if ax is not None else plt.subplots(
            figsize=(10, 6)
        )
        configs = summary.index.tolist()
        latency_values = summary["avg_latency_s"].values

        bars = ax_.bar(configs, latency_values, color=_PALETTE[: len(configs)])
        ax_.set_ylabel("Average Latency (seconds)")
        ax_.set_title("Average Query Latency by Defense Configuration")
        ax_.set_xticklabels(configs, rotation=30, ha="right")

        for bar, val in zip(bars, latency_values):
            ax_.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2f}s",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # Add overhead ratio relative to baseline
        baseline_latency = summary.loc["none", "avg_latency_s"] if "none" in summary.index else None
        if baseline_latency and baseline_latency > 0:
            ax2 = ax_.twinx()
            overhead_values = [v / baseline_latency for v in latency_values]
            ax2.plot(
                configs,
                overhead_values,
                "ro--",
                label="Overhead ratio",
                markersize=6,
            )
            ax2.set_ylabel("Latency Overhead (x baseline)")
            ax2.legend(loc="upper left")

        fig.tight_layout()
        return fig

    def plot_threshold_sweep(
        self,
        metric_fn: Callable[[pd.DataFrame], float],
        param_name: str,
        param_values: list[float],
        pipeline_factory: Callable | None = None,
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Sweep a defense threshold parameter and plot the resulting metric.

        If *pipeline_factory* is None, this method operates on the existing
        results DataFrame by filtering on pre-computed threshold values
        stored in the ``details`` column.  Otherwise, it re-runs the
        evaluation for each parameter value (expensive).

        For lightweight usage without re-running the pipeline, the caller
        can pre-compute results at various thresholds and pass a
        *metric_fn* that extracts the desired metric from a filtered
        DataFrame.

        Parameters
        ----------
        metric_fn:
            A callable ``(df_subset) -> float`` that computes the metric
            from a results subset.
        param_name:
            Human-readable name of the parameter being swept (for labels).
        param_values:
            The parameter values to sweep over.
        pipeline_factory:
            Optional factory for re-running evaluations.  Not used in the
            lightweight mode.
        ax:
            Optional axes to plot on.

        Returns
        -------
        plt.Figure
        """
        fig, ax_ = (ax.figure, ax) if ax is not None else plt.subplots(
            figsize=(10, 6)
        )

        metric_values: list[float] = []
        for val in param_values:
            # Lightweight mode: user provides a metric_fn that can handle
            # the threshold value as a filter on existing data.
            try:
                m = metric_fn(self._df, val)  # type: ignore[call-arg]
            except TypeError:
                # Fallback: metric_fn only takes the DataFrame
                m = metric_fn(self._df)
            metric_values.append(m)

        ax_.plot(param_values, metric_values, "b-o", linewidth=2, markersize=6)
        ax_.set_xlabel(param_name)
        ax_.set_ylabel("Metric Value")
        ax_.set_title(f"Threshold Sweep: {param_name}")
        ax_.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Full report generation
    # ------------------------------------------------------------------

    def generate_full_report(self, output_dir: str | Path) -> None:
        """Save all charts and the summary CSV to a directory.

        Creates the following files in *output_dir*:
        - ``summary.csv`` — pivot table
        - ``asr_comparison.png`` — ASR bar chart
        - ``defense_heatmap.png`` — block rate heatmap
        - ``fpr_comparison.png`` — FPR bar chart
        - ``latency_comparison.png`` — latency bar chart
        - ``results.csv`` — full raw results

        Parameters
        ----------
        output_dir:
            Directory to write report artifacts into.  Created if it
            does not exist.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Summary CSV
        summary = self.summary_table()
        summary.to_csv(out / "summary.csv")
        print(f"  Saved summary table -> {out / 'summary.csv'}")

        # Full raw results
        # Drop columns that are not JSON-serializable before saving
        export_df = self._df.drop(
            columns=["details", "retrieved_node_ids"],
            errors="ignore",
        )
        export_df.to_csv(out / "results.csv", index=False)
        print(f"  Saved raw results  -> {out / 'results.csv'}")

        # ASR comparison
        fig = self.plot_asr_comparison()
        self._save_fig(fig, out / "asr_comparison.png")
        print(f"  Saved chart        -> {out / 'asr_comparison.png'}")

        # Defense heatmap
        fig = self.plot_defense_heatmap()
        self._save_fig(fig, out / "defense_heatmap.png")
        print(f"  Saved chart        -> {out / 'defense_heatmap.png'}")

        # FPR comparison
        fig = self.plot_fpr_comparison()
        self._save_fig(fig, out / "fpr_comparison.png")
        print(f"  Saved chart        -> {out / 'fpr_comparison.png'}")

        # Latency comparison
        fig = self.plot_latency_comparison()
        self._save_fig(fig, out / "latency_comparison.png")
        print(f"  Saved chart        -> {out / 'latency_comparison.png'}")

        print(f"\nFull report written to {out}/")
