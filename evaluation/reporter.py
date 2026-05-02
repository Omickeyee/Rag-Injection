from __future__ import annotations
from pathlib import Path
from typing import Any, Callable
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from evaluation.metrics import attack_success_rate, defense_block_rate, false_positive_rate, latency_overhead

_PALETTE = sns.color_palette("Set2", 8)

class EvaluationReporter:
    def __init__(self, results_df):
        self._df = results_df
        
    @property
    def _attack_df(self):
        return self._df[self._df["query_type"] != "benign"]

    @property
    def _benign_df(self):
        return self._df[self._df["query_type"] == "benign"]

    @property
    def _config_names(self):
        return list(self._df["config_name"].unique())

    @property
    def _attack_types(self):
        return list(self._attack_df["query_type"].unique())

    @staticmethod
    def _save_fig(fig, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        
    def summary_table(self):
        records = []
        for config in self._config_names:
            cfg_attack = self._attack_df[
                self._attack_df["config_name"] == config
            ]
            row = {"config_name": config}
            for atype in self._attack_types:
                subset = cfg_attack[cfg_attack["query_type"] == atype]
                asr = attack_success_rate(subset.to_dict("records"))
                row[atype] = round(asr, 4)
            overall_asr = attack_success_rate(cfg_attack.to_dict("records"))
            row["overall"] = round(overall_asr, 4)
            cfg_benign = self._benign_df[self._benign_df["config_name"] == config]
            fpr = false_positive_rate(cfg_benign.to_dict("records"))
            row["fpr"] = round(fpr, 4)
            avg_latency = cfg_attack["timing_total"].mean()
            row["avg_latency_s"] = round(avg_latency, 3) if pd.notna(avg_latency) else 0.0
            records.append(row)
        return pd.DataFrame(records).set_index("config_name")
    
    def plot_asr_comparison(self, ax: plt.Axes | None = None) -> plt.Figure:
        summary = self.summary_table()
        fig, ax_ = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(10, 6))
        configs = summary.index.tolist()
        asr_values = summary["overall"].values
        bars = ax_.bar(configs, asr_values, color=_PALETTE[: len(configs)])
        ax_.set_ylabel("Attack Success Rate")
        ax_.set_title("Attack Success Rate by Defense Configuration")
        ax_.set_ylim(0, 1.05)
        ax_.set_xticklabels(configs, rotation=30, ha="right")
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

    def plot_defense_heatmap(self, ax = None):
        configs = self._config_names
        attack_types = self._attack_types
        none_attack = self._attack_df[self._attack_df["config_name"] == "none"]
        none_results_by_type = {}
        for atype in attack_types:
            subset = none_attack[none_attack["query_type"] == atype]
            none_results_by_type[atype] = subset.to_dict("records")
        matrix = []
        row_labels = []
        for config in configs:
            if config == "none":
                continue
            row_labels.append(config)
            row = []
            cfg_attack = self._attack_df[self._attack_df["config_name"] == config]
            for atype in attack_types:
                defense_subset = cfg_attack[
                    cfg_attack["query_type"] == atype
                ].to_dict("records")
                baseline_subset = none_results_by_type.get(atype, [])
                br = defense_block_rate(baseline_subset, defense_subset)
                row.append(round(br, 4))
            matrix.append(row)
        heatmap_df = pd.DataFrame(matrix, index=row_labels, columns=attack_types)
        fig, ax_ = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(10, 6))
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

    def plot_fpr_comparison(self, ax = None):
        summary = self.summary_table()
        fig, ax_ = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(10, 6))
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

    def plot_latency_comparison(self, ax = None):
        summary = self.summary_table()
        fig, ax_ = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(10, 6))
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

    def plot_threshold_sweep(self, metric_fn, param_name, param_values, pipeline_factory = None, ax = None):
        fig, ax_ = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(10, 6))
        metric_values = []
        for val in param_values:
            try:
                m = metric_fn(self._df, val)
            except TypeError:
                m = metric_fn(self._df)
            metric_values.append(m)
        ax_.plot(param_values, metric_values, "b-o", linewidth=2, markersize=6)
        ax_.set_xlabel(param_name)
        ax_.set_ylabel("Metric Value")
        ax_.set_title(f"Threshold Sweep: {param_name}")
        ax_.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig
    
    def generate_full_report(self, output_dir):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        summary = self.summary_table()
        summary.to_csv(out / "summary.csv")
        print(f"\t-Saved summary table to {out / 'summary.csv'}")
        export_df = self._df.drop(
            columns=["details", "retrieved_node_ids"],
            errors="ignore",
        )
        export_df.to_csv(out / "results.csv", index=False)
        print(f"\t-Saved raw results to {out / 'results.csv'}")
        fig = self.plot_asr_comparison()
        self._save_fig(fig, out / "asr_comparison.png")
        print(f"\t-Saved chart to {out / 'asr_comparison.png'}")
        fig = self.plot_defense_heatmap()
        self._save_fig(fig, out / "defense_heatmap.png")
        print(f"\t-Saved chart to {out / 'defense_heatmap.png'}")
        fig = self.plot_fpr_comparison()
        self._save_fig(fig, out / "fpr_comparison.png")
        print(f"\t-Saved chart to {out / 'fpr_comparison.png'}")
        fig = self.plot_latency_comparison()
        self._save_fig(fig, out / "latency_comparison.png")
        print(f"\t-Saved chart to {out / 'latency_comparison.png'}")
        print(f"\nFull report written to {out}/")
