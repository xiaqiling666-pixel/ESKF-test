from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..config import AppConfig, project_path
from .contract import ensure_standard_sensor_dataframe
from .input_quality import build_input_quality_report


@dataclass(frozen=True)
class NavigationReferenceOverride:
    reference_lat_deg: float
    reference_lon_deg: float
    reference_height_m: float


@dataclass(frozen=True)
class DatasetLoadResult:
    dataframe: pd.DataFrame
    navigation_reference_override: NavigationReferenceOverride | None = None
    source_summary: dict[str, str] | None = None
    input_quality_metrics: dict[str, float] | None = None


def load_dataset_from_config(config: AppConfig) -> DatasetLoadResult:
    adapter_kind = config.dataset_adapter.kind

    if adapter_kind == "standard_csv":
        dataset_path = project_path(config.dataset_path)
        result = DatasetLoadResult(
            dataframe=pd.read_csv(dataset_path),
            source_summary={
                "adapter_kind": "standard_csv",
                "filter_input_dataset": str(config.dataset_path),
            },
        )
    elif adapter_kind == "great_msf_imu_ins":
        from .great_msf_dataset import load_great_msf_imu_ins_dataset

        result = load_great_msf_imu_ins_dataset(config)
    elif adapter_kind == "dx_decoded_flight_csv":
        from .dx_decoded_dataset import load_dx_decoded_flight_dataset

        result = load_dx_decoded_flight_dataset(config)
    elif adapter_kind == "dx_imu_external_solution":
        from .dx_external_solution_dataset import load_dx_imu_external_solution_dataset

        result = load_dx_imu_external_solution_dataset(config)
    else:
        raise ValueError(f"不支持的数据适配器类型: {adapter_kind}")

    standardized_dataframe = ensure_standard_sensor_dataframe(
        result.dataframe,
        source_name=f"adapter:{adapter_kind}",
    )
    input_quality_report = build_input_quality_report(result.dataframe, standardized_dataframe)
    source_summary = dict(result.source_summary or {})
    source_summary.update(input_quality_report.summary)
    return DatasetLoadResult(
        dataframe=standardized_dataframe,
        navigation_reference_override=result.navigation_reference_override,
        source_summary=source_summary,
        input_quality_metrics=input_quality_report.metrics,
    )


def load_dataframe_from_config(config: AppConfig) -> pd.DataFrame:
    return load_dataset_from_config(config).dataframe
