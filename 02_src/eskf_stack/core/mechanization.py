from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .math_utils import quat_multiply, quat_normalize, quat_to_rotmat, rotvec_to_quat
from .navigation import LocalNavigationEnvironment, resolve_local_navigation_environment
from .state import NavState


@dataclass(frozen=True)
class CorrectedImu:
    accel_body: np.ndarray
    gyro_body: np.ndarray


@dataclass(frozen=True)
class MechanizationArtifacts:
    corrected_imu: CorrectedImu
    body_rate_wrt_nav: np.ndarray
    navigation_environment: LocalNavigationEnvironment
    position_mid_nav: np.ndarray
    velocity_mid_nav: np.ndarray
    rot_mid: np.ndarray
    accel_nav: np.ndarray
    specific_force_nav: np.ndarray
    coriolis_nav: np.ndarray


def correct_imu_measurement(state: NavState, accel_meas: np.ndarray, gyro_meas: np.ndarray) -> CorrectedImu:
    return CorrectedImu(
        accel_body=accel_meas - state.accel_bias,
        gyro_body=gyro_meas - state.gyro_bias,
    )


def attitude_update(state: NavState, corrected_gyro: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    quat_prev = state.quaternion
    half_delta_quat = rotvec_to_quat(corrected_gyro * dt * 0.5)
    quat_mid = quat_normalize(quat_multiply(quat_prev, half_delta_quat))
    quat_cur = quat_normalize(quat_multiply(quat_prev, rotvec_to_quat(corrected_gyro * dt)))
    return quat_cur, quat_to_rotmat(quat_mid)


def velocity_update(
    state: NavState,
    corrected_accel: np.ndarray,
    rot_mid: np.ndarray,
    dt: float,
    environment: LocalNavigationEnvironment,
    velocity_for_coriolis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    specific_force_nav = rot_mid @ corrected_accel
    coriolis_nav = -np.cross(environment.omega_coriolis_nav, velocity_for_coriolis)
    accel_nav = specific_force_nav + environment.gravity_vector + coriolis_nav
    velocity_cur = state.velocity + accel_nav * dt
    return velocity_cur, accel_nav, specific_force_nav, coriolis_nav


def _nav_frame_rate_in_body(state: NavState, environment: LocalNavigationEnvironment) -> np.ndarray:
    return state.rotation.T @ environment.omega_in_nav


def _nav_frame_rate_in_body_from_rot(rot_nav_from_body: np.ndarray, environment: LocalNavigationEnvironment) -> np.ndarray:
    return rot_nav_from_body.T @ environment.omega_in_nav


def position_update(state: NavState, velocity_cur: np.ndarray, dt: float) -> np.ndarray:
    return state.position + 0.5 * (state.velocity + velocity_cur) * dt


def mechanize_local_frame(
    state: NavState,
    accel_meas: np.ndarray,
    gyro_meas: np.ndarray,
    dt: float,
    base_environment: LocalNavigationEnvironment,
    use_wgs84_gravity: bool,
    use_earth_rotation: bool,
) -> MechanizationArtifacts:
    corrected_imu = correct_imu_measurement(state, accel_meas, gyro_meas)

    pre_environment = resolve_local_navigation_environment(
        base_environment,
        state.position,
        state.velocity,
        use_wgs84_gravity=use_wgs84_gravity,
        use_earth_rotation=use_earth_rotation,
    )

    # First use the pre-state environment to predict a midpoint navigation environment.
    nav_rate_in_body_pre = _nav_frame_rate_in_body(state, pre_environment)
    body_rate_pre = corrected_imu.gyro_body - nav_rate_in_body_pre
    _, rot_mid_pre = attitude_update(state, body_rate_pre, dt)
    specific_force_nav_pre = rot_mid_pre @ corrected_imu.accel_body
    coriolis_nav_pre = -np.cross(pre_environment.omega_coriolis_nav, state.velocity)
    accel_nav_pre = specific_force_nav_pre + pre_environment.gravity_vector + coriolis_nav_pre
    velocity_mid = state.velocity + 0.5 * accel_nav_pre * dt
    position_mid = state.position + 0.5 * state.velocity * dt + 0.125 * accel_nav_pre * dt * dt

    mid_environment = resolve_local_navigation_environment(
        base_environment,
        position_mid,
        velocity_mid,
        use_wgs84_gravity=use_wgs84_gravity,
        use_earth_rotation=use_earth_rotation,
    )

    # Then propagate with the midpoint environment for this whole step.
    nav_rate_in_body_mid = _nav_frame_rate_in_body_from_rot(rot_mid_pre, mid_environment)
    body_rate_wrt_nav = corrected_imu.gyro_body - nav_rate_in_body_mid
    quat_cur, rot_mid = attitude_update(state, body_rate_wrt_nav, dt)
    velocity_cur, accel_nav, specific_force_nav, coriolis_nav = velocity_update(
        state,
        corrected_imu.accel_body,
        rot_mid,
        dt,
        mid_environment,
        velocity_mid,
    )
    position_cur = position_update(state, velocity_cur, dt)

    state.position = position_cur
    state.velocity = velocity_cur
    state.quaternion = quat_cur

    return MechanizationArtifacts(
        corrected_imu=corrected_imu,
        body_rate_wrt_nav=body_rate_wrt_nav,
        navigation_environment=mid_environment,
        position_mid_nav=position_mid,
        velocity_mid_nav=velocity_mid,
        rot_mid=rot_mid,
        accel_nav=accel_nav,
        specific_force_nav=specific_force_nav,
        coriolis_nav=coriolis_nav,
    )
