from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core import ImuInitializationSample
from .initialization_controller import (
    assess_initialization_status,
    bootstrap_initialize_from_position_pair,
    initialize_filter,
    preinit_sample_from_frame,
    summarize_initialization_status,
)
from .record_builder import build_preinit_record


@dataclass
class InitializationRuntimeState:
    bootstrap_anchor_frame: Any = None
    initialization_samples: list[ImuInitializationSample] = field(default_factory=list)
    initialization_summary: dict[str, str] = field(
        default_factory=lambda: {
            "initialization_mode": "uninitialized",
            "initialization_phase": "WAITING_GNSS",
            "initialization_reason": "no_gnss_position",
            "initialization_ready_mode": "",
            "heading_source": "none",
            "static_coarse_alignment_used": "false",
            "static_alignment_ready": "false",
            "static_alignment_reason": "not_evaluated",
        }
    )


@dataclass(frozen=True)
class InitializationPassResult:
    runtime_state: InitializationRuntimeState
    initialized_this_frame: bool
    preinit_record: dict[str, Any] | None = None


def run_initialization_pass(
    *,
    filter_engine,
    raw_frame,
    measurement_frame,
    truth_frame,
    runtime_state: InitializationRuntimeState,
) -> InitializationPassResult:
    if filter_engine.initialized:
        return InitializationPassResult(
            runtime_state=runtime_state,
            initialized_this_frame=False,
            preinit_record=None,
        )

    runtime_state.initialization_samples.append(preinit_sample_from_frame(raw_frame))
    initialization_status, _ = assess_initialization_status(
        filter_engine,
        measurement_frame,
        runtime_state.initialization_samples,
        runtime_state.bootstrap_anchor_frame,
    )
    summarize_initialization_status(initialization_status, runtime_state.initialization_summary)

    if initialize_filter(
        filter_engine,
        measurement_frame,
        runtime_state.initialization_samples,
        runtime_state.initialization_summary,
        runtime_state.bootstrap_anchor_frame,
    ):
        return InitializationPassResult(
            runtime_state=runtime_state,
            initialized_this_frame=True,
            preinit_record=None,
        )

    if measurement_frame.gnss_pos is not None:
        if runtime_state.bootstrap_anchor_frame is None:
            runtime_state.bootstrap_anchor_frame = measurement_frame
        elif bootstrap_initialize_from_position_pair(
            filter_engine,
            runtime_state.bootstrap_anchor_frame,
            measurement_frame,
            runtime_state.initialization_samples,
            runtime_state.initialization_summary,
        ):
            return InitializationPassResult(
                runtime_state=runtime_state,
                initialized_this_frame=True,
                preinit_record=None,
            )
        else:
            displacement = measurement_frame.gnss_pos - runtime_state.bootstrap_anchor_frame.gnss_pos
            horizontal_displacement = float((displacement[:2] @ displacement[:2]) ** 0.5)
            if measurement_frame.time - runtime_state.bootstrap_anchor_frame.time > 1.0 and horizontal_displacement < 0.5:
                runtime_state.bootstrap_anchor_frame = measurement_frame

    return InitializationPassResult(
        runtime_state=runtime_state,
        initialized_this_frame=False,
        preinit_record=build_preinit_record(
            measurement_frame,
            truth_frame,
            initialization_status,
            runtime_state.bootstrap_anchor_frame,
        ),
    )
