# robot/motion_math.py — TCP pose math (UR rotation-vector convention)

import math


def rotvec_to_rotmat(rx: float, ry: float, rz: float) -> tuple[tuple[float, float, float], ...]:
    """Axis-angle rotation vector (UR convention) to 3x3 rotation matrix."""
    theta = math.sqrt(rx * rx + ry * ry + rz * rz)
    if theta < 1e-9:
        return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    kx, ky, kz = rx / theta, ry / theta, rz / theta
    c = math.cos(theta)
    s = math.sin(theta)
    v = 1.0 - c
    return (
        (kx * kx * v + c, kx * ky * v - kz * s, kx * kz * v + ky * s),
        (ky * kx * v + kz * s, ky * ky * v + c, ky * kz * v - kx * s),
        (kz * kx * v - ky * s, kz * ky * v + kx * s, kz * kz * v + c),
    )


def mat_vec(
    m: tuple[tuple[float, float, float], ...], v: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    )


def tool_horizontal_unit(rotvec: tuple[float, float, float], tool_axis: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Map a tool-frame unit axis to a world horizontal unit vector (Z component zeroed).
    Keeps table-plane moves aligned with gripper heading instead of base Y only.
    """
    r = rotvec_to_rotmat(*rotvec)
    wx, wy, wz = mat_vec(r, tool_axis)
    wx, wy, wz = wx, wy, 0.0
    norm = math.hypot(wx, wy)
    if norm < 1e-6:
        return (0.0, 0.0, 0.0)
    return (wx / norm, wy / norm, 0.0)
