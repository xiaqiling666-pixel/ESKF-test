from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..config import AppConfig
from .contract import ensure_standard_sensor_dataframe
from .csv_dataset import load_sensor_dataframe


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


def load_dataset_from_config(config: AppConfig) -> DatasetLoadResult:
    adapter_kind = config.dataset_adapter.kind

    if adapter_kind == "standard_csv":
        result = DatasetLoadResult(
            dataframe=load_sensor_dataframe(config.dataset_path),
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
    return DatasetLoadResult(
        dataframe=standardized_dataframe,
        navigation_reference_override=result.navigation_reference_override,
        source_summary=result.source_summary,
    )


def load_dataframe_from_config(config: AppConfig) -> pd.DataFrame:
    return load_dataset_from_config(config).dataframe
