from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..core.math_utils import wrap_angle
from .truth_access import has_diagnostic_truth, truth_position_columns, truth_velocity_columns, truth_yaw_column


plt.style.use("seaborn-v0_8-whitegrid")


def _mode_mapping(*mode_series: pd.Series) -> dict[str, int]:
    ordered_names: list[str] = []
    for series in mode_series:
        if series is None:
            continue
        for value in series:
            if isinstance(value, str) and value and value not in ordered_names:
                ordered_names.append(value)
    return {name: index for index, name in enumerate(ordered_names)}


def _sample_durations(time_series: pd.Series) -> pd.Series:
    if len(time_series) <= 1:
        return pd.Series([0.0] * len(time_series), index=time_series.index, dtype=float)
    durations = time_series.shift(-1) - time_series
    positive_steps = durations[durations > 0.0]
    fallback_step = float(positive_steps.median()) if not positive_steps.empty else 0.0
    durations = durations.fillna(fallback_step).clip(lower=0.0)
    return durations.astype(float)


def _categorical_duration_summary(result_df: pd.DataFrame, category_column: str) -> pd.DataFrame:
    if category_column not in result_df.columns or "time" not in result_df.columns or result_df.empty:
        return pd.DataFrame(columns=["category", "duration_s", "share_pct", "entry_count"])

    category_series = result_df[category_column].fillna("").astype(str)
    category_series = category_series.where(category_series != "", other="EMPTY")
    sample_durations = _sample_durations(result_df["time"])
    total_duration = float(sample_durations.sum())
    enter_flags = category_series.ne(category_series.shift(1))

    rows: list[dict[str, float | str]] = []
    for category_name, _ in result_df.groupby(category_series, sort=False):
        category_mask = category_series == category_name
        duration_s = float(sample_durations[category_mask].sum())
        entry_count = int((category_mask & enter_flags).sum())
        share_pct = 0.0 if total_duration <= 0.0 else 100.0 * duration_s / total_duration
        rows.append(
            {
                "category": category_name,
                "duration_s": duration_s,
                "share_pct": share_pct,
                "entry_count": float(entry_count),
            }
        )
    return pd.DataFrame(rows)


def save_trajectory_plot(result_df: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(result_df["est_x"], result_df["est_y"], label="ESKF")
    if has_diagnostic_truth(result_df):
        truth_position = truth_position_columns()
        axes[0].plot(result_df[truth_position[0]], result_df[truth_position[1]], "--", label="Truth")
    axes[0].set_title("XY trajectory")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].axis("equal")
    axes[0].legend()

    axes[1].plot(result_df["time"], result_df["est_z"], label="ESKF z")
    if has_diagnostic_truth(result_df):
        axes[1].plot(result_df["time"], result_df[truth_position_columns()[2]], "--", label="Truth z")
    axes[1].set_title("Altitude over time")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("z [m]")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_error_plot(result_df: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    if has_diagnostic_truth(result_df):
        position_error = result_df[["est_x", "est_y", "est_z"]].to_numpy() - result_df[list(truth_position_columns())].to_numpy()
        position_norm = np.linalg.norm(position_error, axis=1)
        axes[0].plot(result_df["time"], position_norm)
        axes[0].set_ylabel("pos err [m]")

    if has_diagnostic_truth(result_df):
        velocity_error = result_df[["est_vx", "est_vy", "est_vz"]].to_numpy() - result_df[list(truth_velocity_columns())].to_numpy()
        velocity_norm = np.linalg.norm(velocity_error, axis=1)
        axes[1].plot(result_df["time"], velocity_norm, color="tab:orange")
        axes[1].set_ylabel("vel err [m/s]")

    if has_diagnostic_truth(result_df):
        yaw_error_deg = [
            np.rad2deg(wrap_angle(est - truth))
            for est, truth in zip(result_df["est_yaw"], result_df[truth_yaw_column()], strict=False)
        ]
        axes[2].plot(result_df["time"], yaw_error_deg, color="tab:green")
        axes[2].set_ylabel("yaw err [deg]")

    axes[2].set_xlabel("time [s]")
    axes[0].set_title("Fusion error summary")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_quality_plot(result_df: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

    axes[0].plot(result_df["time"], result_df["quality_score"], color="tab:purple")
    axes[0].set_ylabel("quality")
    axes[0].set_ylim(0.0, 100.0)
    axes[0].set_title("Quality score and mode")
    if "mode_transition_pending" in result_df.columns:
        pending_mask = result_df["mode_transition_pending"].astype(bool)
        if pending_mask.any():
            axes[0].fill_between(
                result_df["time"],
                0.0,
                100.0,
                where=pending_mask,
                color="tab:gray",
                alpha=0.12,
                step="post",
                label="transition pending",
            )
            axes[0].legend(loc="lower left")

    mode_to_index = _mode_mapping(
        result_df["mode"],
        result_df["mode_target"] if "mode_target" in result_df.columns else None,
        result_df["mode_candidate"] if "mode_candidate" in result_df.columns else None,
    )
    mode_index = [mode_to_index[name] for name in result_df["mode"]]
    axes[1].step(result_df["time"], mode_index, where="post", color="tab:red", label="active")
    if "mode_target" in result_df.columns:
        target_index = [mode_to_index[name] for name in result_df["mode_target"]]
        axes[1].step(result_df["time"], target_index, where="post", color="tab:blue", alpha=0.7, label="target")
    axes[1].set_ylabel("mode")
    axes[1].set_yticks(list(mode_to_index.values()))
    axes[1].set_yticklabels(list(mode_to_index.keys()))
    axes[1].legend(loc="upper right")

    if "mode_candidate" in result_df.columns and "mode_candidate_hold_s" in result_df.columns:
        candidate_labels = result_df["mode_candidate"].fillna("").astype(str)
        has_candidate = candidate_labels != ""
        candidate_index = np.array(
            [np.nan if not label else mode_to_index[label] for label in candidate_labels],
            dtype=float,
        )
        axes[2].step(result_df["time"], candidate_index, where="post", color="tab:orange", label="candidate")
        hold_series = result_df["mode_candidate_hold_s"].to_numpy()
        hold_axis = axes[2].twinx()
        hold_axis.plot(result_df["time"], hold_series, color="tab:green", label="candidate hold [s]")
        if has_candidate.any():
            axes[2].fill_between(
                result_df["time"],
                -0.5,
                len(mode_to_index) - 0.5,
                where=has_candidate.to_numpy(),
                color="tab:orange",
                alpha=0.08,
                step="post",
            )
        axes[2].set_ylabel("candidate")
        axes[2].set_yticks(list(mode_to_index.values()))
        axes[2].set_yticklabels(list(mode_to_index.keys()))
        hold_axis.set_ylabel("hold [s]")
        left_handles, left_labels = axes[2].get_legend_handles_labels()
        right_handles, right_labels = hold_axis.get_legend_handles_labels()
        hold_axis.legend(left_handles + right_handles, left_labels + right_labels, loc="upper right")

    axes[2].set_xlabel("time [s]")

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_navigation_plot(result_df: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True)

    if {"nav_lat_deg", "nav_lon_deg", "nav_height_m"}.issubset(result_df.columns):
        axes[0].plot(result_df["time"], result_df["nav_lat_deg"], label="lat [deg]")
        axes[0].plot(result_df["time"], result_df["nav_lon_deg"], label="lon [deg]")
        axes[0].plot(result_df["time"], result_df["nav_height_m"], label="height [m]")
        axes[0].legend()
        axes[0].set_ylabel("geo / h")
        axes[0].set_title("Navigation environment diagnostics")

    if {"earth_rate_n_y", "earth_rate_n_z"}.issubset(result_df.columns):
        axes[1].plot(result_df["time"], result_df["earth_rate_n_y"], label="omega_ie,y")
        axes[1].plot(result_df["time"], result_df["earth_rate_n_z"], label="omega_ie,z")
        axes[1].legend()
        axes[1].set_ylabel("earth rate")

    if {"transport_rate_n_x", "transport_rate_n_y", "transport_rate_n_z"}.issubset(result_df.columns):
        axes[2].plot(result_df["time"], result_df["transport_rate_n_x"], label="omega_en,x")
        axes[2].plot(result_df["time"], result_df["transport_rate_n_y"], label="omega_en,y")
        axes[2].plot(result_df["time"], result_df["transport_rate_n_z"], label="omega_en,z")
        axes[2].legend()
        axes[2].set_ylabel("transport rate")

    gradient_plotted = False
    if "gravity_gradient_norm" in result_df.columns:
        axes[3].semilogy(
            result_df["time"],
            np.clip(result_df["gravity_gradient_norm"], 1e-16, None),
            color="tab:brown",
            label="|dg/dr|",
        )
        gradient_plotted = True
    if "coriolis_position_gradient_norm" in result_df.columns:
        axes[3].semilogy(
            result_df["time"],
            np.clip(result_df["coriolis_position_gradient_norm"], 1e-16, None),
            color="tab:green",
            label="|d coriolis / dr|",
        )
        gradient_plotted = True
    if "coriolis_velocity_gradient_norm" in result_df.columns:
        axes[3].semilogy(
            result_df["time"],
            np.clip(result_df["coriolis_velocity_gradient_norm"], 1e-16, None),
            color="tab:red",
            label="|d coriolis / dv|",
        )
        gradient_plotted = True
    if gradient_plotted:
        axes[3].set_ylabel("grad norm")
        axes[3].legend(loc="upper left")

    if {"meridian_radius_m", "prime_vertical_radius_m"}.issubset(result_df.columns):
        axes[4].plot(
            result_df["time"],
            result_df["meridian_radius_m"] - result_df["meridian_radius_m"].iloc[0],
            "--",
            color="tab:blue",
            label="Rm delta [m]",
        )
        axes[4].plot(
            result_df["time"],
            result_df["prime_vertical_radius_m"] - result_df["prime_vertical_radius_m"].iloc[0],
            "--",
            color="tab:orange",
            label="Rn delta [m]",
        )
        axes[4].set_ylabel("radius delta [m]")
        axes[4].legend(loc="upper right")

    axes[4].set_xlabel("time [s]")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_covariance_plot(result_df: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)

    if {"pos_sigma_norm_m", "vel_sigma_norm_mps", "att_sigma_norm_deg"}.issubset(result_df.columns):
        axes[0].plot(result_df["time"], result_df["pos_sigma_norm_m"], label="pos sigma norm")
        axes[0].set_ylabel("pos sigma [m]")
        axes[0].legend(loc="upper right")

        axes[1].plot(result_df["time"], result_df["vel_sigma_norm_mps"], color="tab:orange", label="vel sigma norm")
        axes[1].set_ylabel("vel sigma [m/s]")
        axes[1].legend(loc="upper right")

        axes[2].plot(result_df["time"], result_df["att_sigma_norm_deg"], color="tab:green", label="att sigma norm")
        axes[2].set_ylabel("att sigma [deg]")
        axes[2].legend(loc="upper right")

    if {"cov_trace", "cov_min_diag"}.issubset(result_df.columns):
        axes[3].plot(result_df["time"], result_df["cov_trace"], color="tab:purple", label="trace(P)")
        axes[3].plot(result_df["time"], result_df["cov_min_diag"], color="tab:red", label="min diag(P)")
        axes[3].set_ylabel("cov health")
        axes[3].legend(loc="upper right")

    axes[0].set_title("Covariance diagnostics")
    axes[3].set_xlabel("time [s]")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_state_machine_summary_plot(result_df: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))

    mode_summary = _categorical_duration_summary(result_df, "mode")
    reason_summary = _categorical_duration_summary(result_df, "mode_reason")

    if not mode_summary.empty:
        axes[0, 0].barh(mode_summary["category"], mode_summary["share_pct"], color="tab:blue")
        axes[0, 0].set_xlabel("share [%]")
        axes[0, 0].set_title("Mode Share")

        axes[0, 1].barh(mode_summary["category"], mode_summary["entry_count"], color="tab:orange")
        axes[0, 1].set_xlabel("entries")
        axes[0, 1].set_title("Mode Entries")

    if not reason_summary.empty:
        top_reason_summary = reason_summary.sort_values("duration_s", ascending=True).tail(6)
        axes[1, 0].barh(top_reason_summary["category"], top_reason_summary["duration_s"], color="tab:green")
        axes[1, 0].set_xlabel("duration [s]")
        axes[1, 0].set_title("Top Reason Durations")

    stats = []
    if "mode_transition_pending" in result_df.columns:
        pending_rows = int(result_df["mode_transition_pending"].astype(bool).sum())
        stats.append(("pending rows", pending_rows))
        if "time" in result_df.columns and len(result_df) > 1:
            time_step = result_df["time"].diff().fillna(0.0)
            pending_duration_s = float(time_step[result_df["mode_transition_pending"].astype(bool)].sum())
            stats.append(("pending s", round(pending_duration_s, 2)))
    if "mode" in result_df.columns:
        mode_series = result_df["mode"].astype(str)
        transition_count = max(int(mode_series.ne(mode_series.shift(1)).sum()) - 1, 0)
        stats.append(("mode transitions", transition_count))
    if "mode_reason" in result_df.columns:
        reason_series = result_df["mode_reason"].astype(str)
        reason_transition_count = max(int(reason_series.ne(reason_series.shift(1)).sum()) - 1, 0)
        stats.append(("reason transitions", reason_transition_count))

    axes[1, 1].axis("off")
    axes[1, 1].set_title("Transition Summary")
    if stats:
        summary_lines = [f"{label}: {value}" for label, value in stats]
        axes[1, 1].text(
            0.02,
            0.98,
            "\n".join(summary_lines),
            va="top",
            ha="left",
            fontsize=11,
            family="monospace",
        )

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
