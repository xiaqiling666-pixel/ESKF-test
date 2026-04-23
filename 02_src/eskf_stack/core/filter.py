from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import AppConfig
from .math_utils import (
    euler_to_quat,
    quat_multiply,
    quat_normalize,
    rotvec_to_quat,
    skew,
)
from .mechanization import mechanize_local_frame
from .navigation import (
    build_navigation_linearization,
    build_local_navigation_environment,
    resolve_local_navigation_environment,
)
from .state import AXIS_DIM, ERROR_STATE, ERROR_STATE_DIM, PROCESS_NOISE, PROCESS_NOISE_DIM, NavState


@dataclass(frozen=True)
class PredictDiagnostics:
    raw_dt: float
    applied_dt: float
    skipped: bool
    warning: bool
    reason: str


class OfflineESKF:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.navigation_environment = build_local_navigation_environment(config)
        self.current_navigation_environment = self.navigation_environment
        self.current_coriolis_position_jacobian = np.zeros((AXIS_DIM, AXIS_DIM))
        self.current_coriolis_velocity_jacobian = np.zeros((AXIS_DIM, AXIS_DIM))
        self.state = NavState.zero()
        self.P = self._build_initial_covariance()
        self.initialized = False
        self.last_predict_diagnostics = PredictDiagnostics(
            raw_dt=0.0,
            applied_dt=0.0,
            skipped=True,
            warning=False,
            reason="not_run",
        )

    def _sync_navigation_environment(self) -> None:
        self.current_navigation_environment = resolve_local_navigation_environment(
            self.navigation_environment,
            self.state.position,
            self.state.velocity,
            use_wgs84_gravity=self.config.navigation_environment.use_wgs84_gravity,
            use_earth_rotation=self.config.navigation_environment.use_earth_rotation,
        )

    def _build_initial_covariance(self) -> np.ndarray:
        init = self.config.initial_covariance
        att_var = np.deg2rad(init.att_std_deg) ** 2
        diagonal = np.zeros(ERROR_STATE_DIM)
        diagonal[ERROR_STATE.position] = init.pos_std**2
        diagonal[ERROR_STATE.velocity] = init.vel_std**2
        diagonal[ERROR_STATE.attitude] = att_var
        diagonal[ERROR_STATE.gyro_bias] = init.gyro_bias_std**2
        diagonal[ERROR_STATE.accel_bias] = init.accel_bias_std**2
        return np.diag(diagonal)

    def initialize(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        yaw: float,
        roll: float = 0.0,
        pitch: float = 0.0,
        gyro_bias: np.ndarray | None = None,
        accel_bias: np.ndarray | None = None,
    ) -> None:
        self.state.position = position.astype(float)
        self.state.velocity = velocity.astype(float)
        self.state.quaternion = euler_to_quat(float(roll), float(pitch), float(yaw))
        self.state.gyro_bias = np.zeros(AXIS_DIM) if gyro_bias is None else gyro_bias.astype(float)
        self.state.accel_bias = np.zeros(AXIS_DIM) if accel_bias is None else accel_bias.astype(float)
        self.P = self._build_initial_covariance()
        self._sync_navigation_environment()
        self.initialized = True

    def _build_continuous_error_jacobian(
        self,
        gravity_gradient_nav: np.ndarray,
        coriolis_position_gradient_nav: np.ndarray,
        coriolis_velocity_gradient_nav: np.ndarray,
        corrected_accel: np.ndarray,
        body_rate_wrt_nav: np.ndarray,
        rot_nav_from_body: np.ndarray,
    ) -> np.ndarray:
        F = np.zeros((ERROR_STATE_DIM, ERROR_STATE_DIM))
        F[ERROR_STATE.position, ERROR_STATE.velocity] = np.eye(AXIS_DIM)
        F[ERROR_STATE.velocity, ERROR_STATE.position] = gravity_gradient_nav + coriolis_position_gradient_nav
        F[ERROR_STATE.velocity, ERROR_STATE.velocity] = coriolis_velocity_gradient_nav
        F[ERROR_STATE.velocity, ERROR_STATE.attitude] = -rot_nav_from_body @ skew(corrected_accel)
        F[ERROR_STATE.velocity, ERROR_STATE.accel_bias] = -rot_nav_from_body
        F[ERROR_STATE.attitude, ERROR_STATE.attitude] = -skew(body_rate_wrt_nav)
        F[ERROR_STATE.attitude, ERROR_STATE.gyro_bias] = -np.eye(AXIS_DIM)
        return F

    def _build_noise_mapping(self, rot_nav_from_body: np.ndarray) -> np.ndarray:
        G = np.zeros((ERROR_STATE_DIM, PROCESS_NOISE_DIM))
        G[ERROR_STATE.velocity, PROCESS_NOISE.accel] = rot_nav_from_body
        G[ERROR_STATE.attitude, PROCESS_NOISE.gyro] = -np.eye(AXIS_DIM)
        G[ERROR_STATE.gyro_bias, PROCESS_NOISE.gyro_bias] = np.eye(AXIS_DIM)
        G[ERROR_STATE.accel_bias, PROCESS_NOISE.accel_bias] = np.eye(AXIS_DIM)
        return G

    def _build_process_noise_covariance(self) -> np.ndarray:
        noise = self.config.process_noise
        diagonal = np.zeros(PROCESS_NOISE_DIM)
        diagonal[PROCESS_NOISE.accel] = noise.accel_std**2
        diagonal[PROCESS_NOISE.gyro] = noise.gyro_std**2
        diagonal[PROCESS_NOISE.gyro_bias] = noise.gyro_bias_std**2
        diagonal[PROCESS_NOISE.accel_bias] = noise.accel_bias_std**2
        return np.diag(diagonal)

    def _discretize_error_model(self, F: np.ndarray, G: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        identity = np.eye(ERROR_STATE_DIM)
        continuous_q = G @ self._build_process_noise_covariance() @ G.T
        phi = identity + F * dt + 0.5 * (F @ F) * (dt * dt)
        qd = continuous_q * dt + 0.5 * (F @ continuous_q + continuous_q @ F.T) * (dt * dt)
        return phi, 0.5 * (qd + qd.T)

    def predict(self, accel_meas: np.ndarray, gyro_meas: np.ndarray, dt: float) -> None:
        time_step_config = self.config.time_step_management
        raw_dt = float(dt)
        applied_dt = raw_dt
        skipped = False
        warning = False
        reason = "applied"

        if raw_dt <= 0.0:
            skipped = True
            applied_dt = 0.0
            reason = "nonpositive_dt"
        elif raw_dt < time_step_config.min_positive_dt_s:
            skipped = True
            applied_dt = 0.0
            warning = True
            reason = "below_min_positive_dt"
        elif raw_dt > time_step_config.max_dt_s:
            warning = True
            if time_step_config.skip_large_dt:
                skipped = True
                applied_dt = 0.0
                reason = "above_max_dt_skipped"
            else:
                applied_dt = time_step_config.max_dt_s
                reason = "above_max_dt_clamped"

        self.last_predict_diagnostics = PredictDiagnostics(
            raw_dt=raw_dt,
            applied_dt=applied_dt,
            skipped=skipped,
            warning=warning,
            reason=reason,
        )

        if skipped:
            self._sync_navigation_environment()
            self.current_coriolis_position_jacobian = np.zeros((AXIS_DIM, AXIS_DIM))
            self.current_coriolis_velocity_jacobian = np.zeros((AXIS_DIM, AXIS_DIM))
            return

        artifacts = mechanize_local_frame(
            self.state,
            accel_meas,
            gyro_meas,
            applied_dt,
            self.navigation_environment,
            use_wgs84_gravity=self.config.navigation_environment.use_wgs84_gravity,
            use_earth_rotation=self.config.navigation_environment.use_earth_rotation,
        )
        self.current_navigation_environment = artifacts.navigation_environment
        corrected_accel = artifacts.corrected_imu.accel_body
        body_rate_wrt_nav = artifacts.body_rate_wrt_nav
        rot_mid = artifacts.rot_mid
        navigation_linearization = build_navigation_linearization(
            self.navigation_environment,
            artifacts.navigation_environment,
            artifacts.position_mid_nav,
            artifacts.velocity_mid_nav,
            use_wgs84_gravity=self.config.navigation_environment.use_wgs84_gravity,
            use_earth_rotation=self.config.navigation_environment.use_earth_rotation,
        )
        self.current_coriolis_position_jacobian = navigation_linearization.coriolis_position_gradient_nav
        self.current_coriolis_velocity_jacobian = navigation_linearization.coriolis_velocity_gradient_nav
        F = self._build_continuous_error_jacobian(
            navigation_linearization.gravity_gradient_nav,
            navigation_linearization.coriolis_position_gradient_nav,
            navigation_linearization.coriolis_velocity_gradient_nav,
            corrected_accel,
            body_rate_wrt_nav,
            rot_mid,
        )
        G = self._build_noise_mapping(rot_mid)
        phi, qd = self._discretize_error_model(F, G, applied_dt)
        self.P = phi @ self.P @ phi.T + qd
        self.P = 0.5 * (self.P + self.P.T)
        self._sync_navigation_environment()

    def _inject_error_state(self, delta_x: np.ndarray) -> None:
        self.state.position += delta_x[ERROR_STATE.position]
        self.state.velocity += delta_x[ERROR_STATE.velocity]
        self.state.quaternion = quat_normalize(
            quat_multiply(self.state.quaternion, rotvec_to_quat(delta_x[ERROR_STATE.attitude]))
        )
        self.state.gyro_bias += delta_x[ERROR_STATE.gyro_bias]
        self.state.accel_bias += delta_x[ERROR_STATE.accel_bias]

    def _apply_reset_jacobian(self, delta_x: np.ndarray) -> None:
        reset_jacobian = np.eye(ERROR_STATE_DIM)
        reset_jacobian[ERROR_STATE.attitude, ERROR_STATE.attitude] -= 0.5 * skew(delta_x[ERROR_STATE.attitude])
        self.P = reset_jacobian @ self.P @ reset_jacobian.T
        self.P = 0.5 * (self.P + self.P.T)

    def apply_linear_update(self, residual: np.ndarray, H: np.ndarray, R: np.ndarray) -> None:
        s_matrix = H @ self.P @ H.T + R
        try:
            kalman_gain = self.P @ H.T @ np.linalg.inv(s_matrix)
        except np.linalg.LinAlgError:
            kalman_gain = self.P @ H.T @ np.linalg.pinv(s_matrix)
        delta_x = kalman_gain @ residual
        identity = np.eye(ERROR_STATE_DIM)
        joseph = identity - kalman_gain @ H
        self.P = joseph @ self.P @ joseph.T + kalman_gain @ R @ kalman_gain.T
        self.P = 0.5 * (self.P + self.P.T)
        self._inject_error_state(delta_x)
        self._apply_reset_jacobian(delta_x)
        self._sync_navigation_environment()
