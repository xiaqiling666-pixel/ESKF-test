from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "01_data" / "config.json"
ALLOWED_CONFIG_PROFILES = {"default_general", "sample_validation", "experiment"}


def project_path(relative_path: str) -> Path:
    return PROJECT_ROOT / relative_path


@dataclass
class ProcessNoise:
    accel_std: float
    gyro_std: float
    gyro_bias_std: float
    accel_bias_std: float


@dataclass
class MeasurementNoise:
    gnss_pos_std: float
    gnss_vel_std: float
    gnss_pos_vertical_std: float
    gnss_vel_vertical_std: float
    baro_std: float
    yaw_std_deg: float


@dataclass
class InitialCovariance:
    pos_std: float
    vel_std: float
    att_std_deg: float
    gyro_bias_std: float
    accel_bias_std: float


@dataclass
class InitializationConfig:
    static_coarse_alignment_enabled: bool
    static_window_duration_s: float
    static_window_min_samples: int
    static_max_accel_std_mps2: float
    static_max_gyro_std_radps: float
    static_gravity_norm_tolerance_mps2: float
    bootstrap_min_dt_s: float
    bootstrap_min_horizontal_displacement_m: float
    heading_wait_timeout_s: float
    zero_yaw_fallback_enabled: bool


@dataclass
class TimeStepManagementConfig:
    min_positive_dt_s: float
    max_dt_s: float
    skip_large_dt: bool


@dataclass
class InnovationManagement:
    gnss_pos_nis_adapt_threshold: float
    gnss_pos_nis_reject_threshold: float
    gnss_vel_nis_adapt_threshold: float
    gnss_vel_nis_reject_threshold: float
    baro_nis_adapt_threshold: float
    baro_nis_reject_threshold: float
    mag_yaw_nis_adapt_threshold: float
    mag_yaw_nis_reject_threshold: float


@dataclass
class FusionPolicy:
    use_nis_rejection: bool
    use_adaptive_r: bool
    use_recovery_scale: bool


@dataclass
class ModeMeasurementScaling:
    enabled: bool
    gnss_pos: dict[str, float] = field(default_factory=dict)
    gnss_vel: dict[str, float] = field(default_factory=dict)


@dataclass
class NavigationEnvironmentConfig:
    frame: str
    reference_lat_deg: float
    reference_lon_deg: float
    reference_height_m: float
    use_wgs84_gravity: bool
    use_earth_rotation: bool


@dataclass
class DatasetAdapterConfig:
    kind: str
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfigMetadata:
    profile: str
    name: str
    purpose: str = ""


@dataclass
class AppConfig:
    config_path: str
    config_metadata: ConfigMetadata
    dataset_path: str
    dataset_adapter: DatasetAdapterConfig
    results_dir: str
    gravity: list[float]
    process_noise: ProcessNoise
    measurement_noise: MeasurementNoise
    initial_covariance: InitialCovariance
    initialization: InitializationConfig
    time_step_management: TimeStepManagementConfig
    innovation_management: InnovationManagement
    fusion_policy: FusionPolicy
    mode_measurement_scaling: ModeMeasurementScaling
    navigation_environment: NavigationEnvironmentConfig
    use_baro: bool
    use_mag: bool


def _load_config_metadata(payload: dict[str, Any], path: Path) -> ConfigMetadata:
    metadata_payload = payload.get("config_metadata")
    if not isinstance(metadata_payload, dict):
        raise ValueError(f"{path} 缺少 config_metadata，无法区分通用配置和样例配置")

    metadata = ConfigMetadata(**metadata_payload)
    if metadata.profile not in ALLOWED_CONFIG_PROFILES:
        raise ValueError(
            f"{path} 的 config_metadata.profile 不合法: {metadata.profile}，"
            f"可选值: {', '.join(sorted(ALLOWED_CONFIG_PROFILES))}"
        )
    if not metadata.name.strip():
        raise ValueError(f"{path} 的 config_metadata.name 不能为空")

    is_default_config = path.resolve() == DEFAULT_CONFIG_PATH.resolve()
    if is_default_config and metadata.profile != "default_general":
        raise ValueError(f"{path} 必须使用 profile=default_general，不能作为样例或实验配置")
    if not is_default_config and metadata.profile == "default_general":
        raise ValueError(f"{path} 不能使用 profile=default_general，这个角色只保留给 01_data/config.json")

    return metadata


def _load_measurement_noise(payload: dict[str, Any]) -> MeasurementNoise:
    measurement_noise_payload = dict(payload["measurement_noise"])
    measurement_noise_payload.setdefault(
        "gnss_pos_vertical_std",
        measurement_noise_payload["gnss_pos_std"],
    )
    measurement_noise_payload.setdefault(
        "gnss_vel_vertical_std",
        measurement_noise_payload["gnss_vel_std"],
    )
    return MeasurementNoise(**measurement_noise_payload)


def load_config(config_path: str | Path | None = None) -> AppConfig:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = _load_config_metadata(payload, path)
    return AppConfig(
        config_path=str(path),
        config_metadata=metadata,
        dataset_path=payload["dataset_path"],
        dataset_adapter=DatasetAdapterConfig(**payload.get("dataset_adapter", {"kind": "standard_csv"})),
        results_dir=payload["results_dir"],
        gravity=payload["gravity"],
        process_noise=ProcessNoise(**payload["process_noise"]),
        measurement_noise=_load_measurement_noise(payload),
        initial_covariance=InitialCovariance(**payload["initial_covariance"]),
        initialization=InitializationConfig(
            **payload.get(
                "initialization",
                {
                    "static_coarse_alignment_enabled": True,
                    "static_window_duration_s": 2.0,
                    "static_window_min_samples": 50,
                    "static_max_accel_std_mps2": 0.25,
                    "static_max_gyro_std_radps": 0.03,
                    "static_gravity_norm_tolerance_mps2": 1.5,
                    "bootstrap_min_dt_s": 0.15,
                    "bootstrap_min_horizontal_displacement_m": 0.5,
                    "heading_wait_timeout_s": 2.0,
                    "zero_yaw_fallback_enabled": False,
                },
            )
        ),
        time_step_management=TimeStepManagementConfig(
            **payload.get(
                "time_step_management",
                {
                    "min_positive_dt_s": 1.0e-4,
                    "max_dt_s": 0.2,
                    "skip_large_dt": True,
                },
            )
        ),
        innovation_management=InnovationManagement(**payload["innovation_management"]),
        fusion_policy=FusionPolicy(
            **payload.get(
                "fusion_policy",
                {
                    "use_nis_rejection": True,
                    "use_adaptive_r": True,
                    "use_recovery_scale": True,
                },
            )
        ),
        mode_measurement_scaling=ModeMeasurementScaling(
            **payload.get(
                "mode_measurement_scaling",
                {
                    "enabled": False,
                    "gnss_pos": {},
                    "gnss_vel": {},
                },
            )
        ),
        navigation_environment=NavigationEnvironmentConfig(
            **payload.get(
                "navigation_environment",
                {
                    "frame": "ENU",
                    "reference_lat_deg": 0.0,
                    "reference_lon_deg": 0.0,
                    "reference_height_m": 0.0,
                    "use_wgs84_gravity": False,
                    "use_earth_rotation": False,
                },
            )
        ),
        use_baro=payload["use_baro"],
        use_mag=payload["use_mag"],
    )
