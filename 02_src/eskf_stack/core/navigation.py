from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..config import AppConfig


WGS84_WIE = 7.2921151467e-5
WGS84_RA = 6378137.0
WGS84_E1 = 0.0066943799901413156


@dataclass(frozen=True)
class LocalNavigationEnvironment:
    frame: str
    reference_lat_rad: float
    reference_lon_rad: float
    reference_height_m: float
    current_lat_rad: float
    current_lon_rad: float
    current_height_m: float
    meridian_radius_m: float
    prime_vertical_radius_m: float
    gravity_vector: np.ndarray
    gravity_gradient_nav: np.ndarray
    earth_rate_nav: np.ndarray
    transport_rate_nav: np.ndarray

    @property
    def omega_in_nav(self) -> np.ndarray:
        return self.earth_rate_nav + self.transport_rate_nav

    @property
    def omega_coriolis_nav(self) -> np.ndarray:
        return 2.0 * self.earth_rate_nav + self.transport_rate_nav


@dataclass(frozen=True)
class NavigationLinearization:
    gravity_gradient_nav: np.ndarray
    coriolis_position_gradient_nav: np.ndarray
    coriolis_velocity_gradient_nav: np.ndarray


def _normal_gravity(latitude_rad: float, height_m: float) -> float:
    sin_lat = np.sin(latitude_rad)
    sin2 = sin_lat * sin_lat
    sin4 = sin2 * sin2
    gamma_a = 9.7803267715
    gamma_0 = gamma_a * (
        1.0
        + 0.0052790414 * sin2
        + 0.0000232718 * sin4
        + 0.0000001262 * sin2 * sin4
        + 0.0000000007 * sin4 * sin4
    )
    return gamma_0 - (3.0877e-6 - 4.3e-9 * sin2) * height_m + 0.72e-12 * height_m * height_m


def _normal_gravity_latitude_derivative(latitude_rad: float, height_m: float) -> float:
    sin_lat = np.sin(latitude_rad)
    cos_lat = np.cos(latitude_rad)
    sin2 = sin_lat * sin_lat
    sin4 = sin2 * sin2
    d_sin2 = 2.0 * sin_lat * cos_lat
    d_sin4 = 2.0 * sin2 * d_sin2
    gamma_a = 9.7803267715
    d_gamma0 = gamma_a * (
        0.0052790414 * d_sin2
        + 0.0000232718 * d_sin4
        + 0.0000001262 * (d_sin2 * sin4 + sin2 * d_sin4)
        + 0.0000000007 * 2.0 * sin4 * d_sin4
    )
    return d_gamma0 + 4.3e-9 * d_sin2 * height_m


def _normal_gravity_height_derivative(latitude_rad: float, height_m: float) -> float:
    sin_lat = np.sin(latitude_rad)
    sin2 = sin_lat * sin_lat
    return -(3.0877e-6 - 4.3e-9 * sin2) + 1.44e-12 * height_m


def _earth_rate_enu(latitude_rad: float) -> np.ndarray:
    return np.array([0.0, WGS84_WIE * np.cos(latitude_rad), WGS84_WIE * np.sin(latitude_rad)])


def _earth_rate_latitude_derivative_enu(latitude_rad: float) -> np.ndarray:
    return np.array([0.0, -WGS84_WIE * np.sin(latitude_rad), WGS84_WIE * np.cos(latitude_rad)])


def _meridian_prime_vertical_radius(latitude_rad: float) -> tuple[float, float]:
    sin_lat = np.sin(latitude_rad)
    tmp = 1.0 - WGS84_E1 * sin_lat * sin_lat
    sqrt_tmp = np.sqrt(tmp)
    meridian_radius = WGS84_RA * (1.0 - WGS84_E1) / (sqrt_tmp * tmp)
    prime_vertical_radius = WGS84_RA / sqrt_tmp
    return meridian_radius, prime_vertical_radius


def _meridian_prime_vertical_radius_derivative(latitude_rad: float) -> tuple[float, float]:
    sin_lat = np.sin(latitude_rad)
    cos_lat = np.cos(latitude_rad)
    tmp = 1.0 - WGS84_E1 * sin_lat * sin_lat
    sqrt_tmp = np.sqrt(tmp)
    common = WGS84_E1 * sin_lat * cos_lat
    meridian_radius_derivative = 3.0 * WGS84_RA * (1.0 - WGS84_E1) * common / (sqrt_tmp * tmp * tmp)
    prime_vertical_radius_derivative = WGS84_RA * common / (sqrt_tmp * tmp)
    return meridian_radius_derivative, prime_vertical_radius_derivative


def _approximate_local_geodetic_state(
    environment: LocalNavigationEnvironment, position_nav: np.ndarray
) -> tuple[float, float, float]:
    east = position_nav[0]
    ref_lat = environment.reference_lat_rad
    ref_lon = environment.reference_lon_rad
    north = position_nav[1]
    up = position_nav[2]
    current_height = environment.reference_height_m + up
    meridian_radius, _ = _meridian_prime_vertical_radius(ref_lat)
    current_lat = ref_lat + north / max(meridian_radius + current_height, 1.0)
    meridian_radius, _ = _meridian_prime_vertical_radius(current_lat)
    current_lat = ref_lat + north / max(meridian_radius + current_height, 1.0)
    _, prime_vertical_radius = _meridian_prime_vertical_radius(current_lat)
    current_lon = ref_lon + east / max(
        (prime_vertical_radius + current_height) * np.cos(current_lat),
        1.0,
    )
    return current_lat, current_lon, current_height


def _local_latitude_height_jacobian(
    base_environment: LocalNavigationEnvironment,
    position_nav: np.ndarray,
    latitude_rad: float,
    height_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    north = position_nav[1]
    meridian_radius, _ = _meridian_prime_vertical_radius(latitude_rad)
    denom = max(meridian_radius + height_m, 1.0)
    latitude_gradient = np.array([0.0, 1.0 / denom, -north / (denom * denom)])
    height_gradient = np.array([0.0, 0.0, 1.0])
    return latitude_gradient, height_gradient


def _transport_rate_enu(latitude_rad: float, height_m: float, velocity_nav: np.ndarray) -> np.ndarray:
    ve = velocity_nav[0]
    vn = velocity_nav[1]
    meridian_radius, prime_vertical_radius = _meridian_prime_vertical_radius(latitude_rad)
    denom_meridian = max(meridian_radius + height_m, 1.0)
    denom_prime_vertical = max(prime_vertical_radius + height_m, 1.0)
    return np.array(
        [
            -vn / denom_meridian,
            ve / denom_prime_vertical,
            ve * np.tan(latitude_rad) / denom_prime_vertical,
        ]
    )


def _transport_rate_velocity_matrix(latitude_rad: float, height_m: float) -> np.ndarray:
    meridian_radius, prime_vertical_radius = _meridian_prime_vertical_radius(latitude_rad)
    denom_meridian = max(meridian_radius + height_m, 1.0)
    denom_prime_vertical = max(prime_vertical_radius + height_m, 1.0)
    return np.array(
        [
            [0.0, -1.0 / denom_meridian, 0.0],
            [1.0 / denom_prime_vertical, 0.0, 0.0],
            [np.tan(latitude_rad) / denom_prime_vertical, 0.0, 0.0],
        ]
    )


def _transport_rate_latitude_derivative(
    latitude_rad: float,
    height_m: float,
    velocity_nav: np.ndarray,
) -> np.ndarray:
    ve = velocity_nav[0]
    vn = velocity_nav[1]
    meridian_radius, prime_vertical_radius = _meridian_prime_vertical_radius(latitude_rad)
    meridian_radius_derivative, prime_vertical_radius_derivative = _meridian_prime_vertical_radius_derivative(
        latitude_rad
    )
    denom_meridian = max(meridian_radius + height_m, 1.0)
    denom_prime_vertical = max(prime_vertical_radius + height_m, 1.0)
    tan_lat = np.tan(latitude_rad)
    sec2_lat = 1.0 / (np.cos(latitude_rad) ** 2)
    return np.array(
        [
            vn * meridian_radius_derivative / (denom_meridian * denom_meridian),
            -ve * prime_vertical_radius_derivative / (denom_prime_vertical * denom_prime_vertical),
            ve
            * (
                sec2_lat / denom_prime_vertical
                - tan_lat * prime_vertical_radius_derivative / (denom_prime_vertical * denom_prime_vertical)
            ),
        ]
    )


def _transport_rate_height_derivative(
    latitude_rad: float,
    height_m: float,
    velocity_nav: np.ndarray,
    step_m: float = 0.1,
) -> np.ndarray:
    rate_plus = _transport_rate_enu(latitude_rad, height_m + step_m, velocity_nav)
    rate_minus = _transport_rate_enu(latitude_rad, height_m - step_m, velocity_nav)
    return (rate_plus - rate_minus) / (2.0 * step_m)


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ]
    )


def _gravity_vector_from_geodetic(
    base_environment: LocalNavigationEnvironment,
    latitude_rad: float,
    height_m: float,
    use_wgs84_gravity: bool,
) -> np.ndarray:
    if not use_wgs84_gravity:
        return base_environment.gravity_vector
    return np.array([0.0, 0.0, -_normal_gravity(latitude_rad, height_m)])


def _gravity_gradient_enu(
    base_environment: LocalNavigationEnvironment,
    position_nav: np.ndarray,
    use_wgs84_gravity: bool,
) -> np.ndarray:
    if not use_wgs84_gravity:
        return np.zeros((3, 3))

    current_lat, _, current_height = _approximate_local_geodetic_state(base_environment, position_nav)
    latitude_gradient, height_gradient = _local_latitude_height_jacobian(
        base_environment,
        position_nav,
        current_lat,
        current_height,
    )
    gravity_latitude_derivative = _normal_gravity_latitude_derivative(current_lat, current_height)
    gravity_height_derivative = _normal_gravity_height_derivative(current_lat, current_height)

    gradient = np.zeros((3, 3))
    gradient[2, :] = -(
        gravity_latitude_derivative * latitude_gradient + gravity_height_derivative * height_gradient
    )
    return gradient


def coriolis_velocity_jacobian(
    environment: LocalNavigationEnvironment,
    velocity_nav: np.ndarray,
) -> np.ndarray:
    transport_rate_velocity_matrix = _transport_rate_velocity_matrix(
        environment.current_lat_rad,
        environment.current_height_m,
    )
    return -_skew(environment.omega_coriolis_nav) + _skew(velocity_nav) @ transport_rate_velocity_matrix


def coriolis_position_jacobian(
    base_environment: LocalNavigationEnvironment,
    position_nav: np.ndarray,
    velocity_nav: np.ndarray,
    use_wgs84_gravity: bool,
    use_earth_rotation: bool,
) -> np.ndarray:
    current_lat, _, current_height = _approximate_local_geodetic_state(base_environment, position_nav)
    latitude_gradient, height_gradient = _local_latitude_height_jacobian(
        base_environment,
        position_nav,
        current_lat,
        current_height,
    )
    omega_latitude_derivative = np.zeros(3)
    if use_earth_rotation:
        omega_latitude_derivative += 2.0 * _earth_rate_latitude_derivative_enu(current_lat)
    omega_latitude_derivative += _transport_rate_latitude_derivative(current_lat, current_height, velocity_nav)
    omega_height_derivative = _transport_rate_height_derivative(current_lat, current_height, velocity_nav)
    omega_position_gradient = np.outer(omega_latitude_derivative, latitude_gradient) + np.outer(
        omega_height_derivative,
        height_gradient,
    )
    return _skew(velocity_nav) @ omega_position_gradient


def build_navigation_linearization(
    base_environment: LocalNavigationEnvironment,
    resolved_environment: LocalNavigationEnvironment,
    position_nav: np.ndarray,
    velocity_nav: np.ndarray,
    use_wgs84_gravity: bool,
    use_earth_rotation: bool,
) -> NavigationLinearization:
    return NavigationLinearization(
        gravity_gradient_nav=resolved_environment.gravity_gradient_nav,
        coriolis_position_gradient_nav=coriolis_position_jacobian(
            base_environment,
            position_nav,
            velocity_nav,
            use_wgs84_gravity=use_wgs84_gravity,
            use_earth_rotation=use_earth_rotation,
        ),
        coriolis_velocity_gradient_nav=coriolis_velocity_jacobian(
            resolved_environment,
            velocity_nav,
        ),
    )


def resolve_local_navigation_environment(
    base_environment: LocalNavigationEnvironment,
    position_nav: np.ndarray,
    velocity_nav: np.ndarray,
    use_wgs84_gravity: bool,
    use_earth_rotation: bool,
) -> LocalNavigationEnvironment:
    current_lat, current_lon, current_height = _approximate_local_geodetic_state(base_environment, position_nav)
    meridian_radius, prime_vertical_radius = _meridian_prime_vertical_radius(current_lat)
    gravity_vector = _gravity_vector_from_geodetic(
        base_environment,
        current_lat,
        current_height,
        use_wgs84_gravity=use_wgs84_gravity,
    )
    gravity_gradient_nav = _gravity_gradient_enu(
        base_environment,
        position_nav,
        use_wgs84_gravity=use_wgs84_gravity,
    )
    earth_rate_nav = _earth_rate_enu(current_lat) if use_earth_rotation else np.zeros(3)
    transport_rate_nav = _transport_rate_enu(current_lat, current_height, velocity_nav)
    return replace(
        base_environment,
        current_lat_rad=current_lat,
        current_lon_rad=current_lon,
        current_height_m=current_height,
        meridian_radius_m=meridian_radius,
        prime_vertical_radius_m=prime_vertical_radius,
        gravity_vector=gravity_vector,
        gravity_gradient_nav=gravity_gradient_nav,
        earth_rate_nav=earth_rate_nav,
        transport_rate_nav=transport_rate_nav,
    )


def build_local_navigation_environment(config: AppConfig) -> LocalNavigationEnvironment:
    nav_config = config.navigation_environment
    frame = nav_config.frame.upper()
    if frame != "ENU":
        raise ValueError(f"当前只支持局部 ENU 导航坐标，收到: {nav_config.frame}")

    latitude_rad = np.deg2rad(nav_config.reference_lat_deg)
    height_m = float(nav_config.reference_height_m)

    if nav_config.use_wgs84_gravity:
        gravity_magnitude = _normal_gravity(latitude_rad, height_m)
        gravity_vector = np.array([0.0, 0.0, -gravity_magnitude])
    else:
        gravity_vector = np.asarray(config.gravity, dtype=float)

    earth_rate_nav = _earth_rate_enu(latitude_rad) if nav_config.use_earth_rotation else np.zeros(3)
    transport_rate_nav = np.zeros(3)

    return LocalNavigationEnvironment(
        frame=frame,
        reference_lat_rad=latitude_rad,
        reference_lon_rad=np.deg2rad(nav_config.reference_lon_deg),
        reference_height_m=height_m,
        current_lat_rad=latitude_rad,
        current_lon_rad=np.deg2rad(nav_config.reference_lon_deg),
        current_height_m=height_m,
        meridian_radius_m=_meridian_prime_vertical_radius(latitude_rad)[0],
        prime_vertical_radius_m=_meridian_prime_vertical_radius(latitude_rad)[1],
        gravity_vector=gravity_vector,
        gravity_gradient_nav=np.zeros((3, 3)),
        earth_rate_nav=earth_rate_nav,
        transport_rate_nav=transport_rate_nav,
    )
