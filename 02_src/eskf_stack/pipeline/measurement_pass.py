from __future__ import annotations

from dataclasses import dataclass

from .record_builder import MeasurementTraceCollector, MeasurementTraceSnapshot


@dataclass(frozen=True)
class MeasurementPassResult:
    trace: MeasurementTraceSnapshot
    innovation_norms: dict[str, float]


def run_measurement_pass(
    *,
    filter_engine,
    measurement_manager,
    freshness_tracker,
    measurement_models,
    measurement_frame,
    current_mode: str | None = None,
) -> MeasurementPassResult:
    trace_collector = MeasurementTraceCollector.create_empty()

    for model in measurement_models:
        policy = model.policy(filter_engine)
        trace_collector.record_policy(model.name, policy)
        result = measurement_manager.process(filter_engine, model, measurement_frame, current_mode=current_mode)
        trace_collector.record_result(result)
        freshness_tracker.note_result(
            result.name,
            measurement_frame.time,
            available=result.available,
            used=result.used,
            rejected=result.rejected,
            reject_bypassed=result.reject_bypassed,
            management_mode=result.management_mode,
            adaptation_scale=result.adaptation_scale,
        )

    return MeasurementPassResult(
        trace=trace_collector.snapshot(),
        innovation_norms=trace_collector.innovation_norms(),
    )
