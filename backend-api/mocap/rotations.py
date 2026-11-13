"""Convert 3D joint positions into per-joint rotations for the BVH skeleton.

MediaPipe only gives joint POSITIONS. A game skeleton needs ROTATIONS. For each bone we
rotate its T-pose direction onto the observed direction (shortest arc, so no twist about
the bone axis). The torso and head have enough landmarks to build a full orientation frame.
"""
from typing import Dict

import numpy as np
from scipy.spatial.transform import Rotation as R

from . import landmarks as L
from .skeleton import CHILDREN, END_SITES, JOINT_BY_NAME, JOINT_ORDER


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-9)


def key_positions(P: np.ndarray) -> Dict[str, np.ndarray]:
    """P: (T, 33, 3) Y-up metres  ->  named points, each (T, 3)."""
    return {
        "hip_c": (P[:, L.L_HIP] + P[:, L.R_HIP]) / 2,
        "sh_c": (P[:, L.L_SHOULDER] + P[:, L.R_SHOULDER]) / 2,
        "ear_c": (P[:, L.L_EAR] + P[:, L.R_EAR]) / 2,
        "nose": P[:, L.NOSE],
        "L_ear": P[:, L.L_EAR], "R_ear": P[:, L.R_EAR],
        "L_hip": P[:, L.L_HIP], "R_hip": P[:, L.R_HIP],
        "L_sh": P[:, L.L_SHOULDER], "R_sh": P[:, L.R_SHOULDER],
        "L_elb": P[:, L.L_ELBOW], "R_elb": P[:, L.R_ELBOW],
        "L_wri": P[:, L.L_WRIST], "R_wri": P[:, L.R_WRIST],
        "L_hand": (P[:, L.L_PINKY] + P[:, L.L_INDEX]) / 2,
        "R_hand": (P[:, L.R_PINKY] + P[:, L.R_INDEX]) / 2,
        "L_knee": P[:, L.L_KNEE], "R_knee": P[:, L.R_KNEE],
        "L_ank": P[:, L.L_ANKLE], "R_ank": P[:, L.R_ANKLE],
        "L_toe": P[:, L.L_FOOT_INDEX], "R_toe": P[:, L.R_FOOT_INDEX],
    }


def frame_from_x_up(x_axis: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Right-handed orthonormal frame (T, 3, 3), columns = (x, y, z), from a side axis and an up hint."""
    x = _unit(x_axis)
    z = _unit(np.cross(x, up))
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=-1)


def head_frame(k: Dict[str, np.ndarray]) -> np.ndarray:
    x = _unit(k["L_ear"] - k["R_ear"])
    forward = k["nose"] - k["ear_c"]
    y = _unit(np.cross(forward, x))
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=-1)


def rotation_between(rest_dir: np.ndarray, target: np.ndarray) -> R:
    """Shortest-arc rotations taking the fixed vector `rest_dir` onto each row of `target` (T, 3)."""
    a = rest_dir / np.linalg.norm(rest_dir)
    b = _unit(target)
    cross = np.cross(np.broadcast_to(a, b.shape), b)
    dot = b @ a
    sin = np.linalg.norm(cross, axis=-1)
    angle = np.arctan2(sin, dot)
    axis = np.zeros_like(b)
    ok = sin > 1e-8
    axis[ok] = cross[ok] / sin[ok, None]
    anti = (~ok) & (dot < 0)                      # exactly opposite: any perpendicular axis works
    if anti.any():
        perp = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(a, [0.0, 1.0, 0.0])
        axis[anti] = perp / np.linalg.norm(perp)
    return R.from_rotvec(axis * angle[:, None])


# joint -> (start point, end point) of the observed bone that joint must point along
BONES = {
    "LeftArm": ("L_sh", "L_elb"), "LeftForeArm": ("L_elb", "L_wri"), "LeftHand": ("L_wri", "L_hand"),
    "RightArm": ("R_sh", "R_elb"), "RightForeArm": ("R_elb", "R_wri"), "RightHand": ("R_wri", "R_hand"),
    "LeftUpLeg": ("L_hip", "L_knee"), "LeftLeg": ("L_knee", "L_ank"), "LeftFoot": ("L_ank", "L_toe"),
    "RightUpLeg": ("R_hip", "R_knee"), "RightLeg": ("R_knee", "R_ank"), "RightFoot": ("R_ank", "R_toe"),
}


def _rest_direction(joint_name: str) -> np.ndarray:
    """T-pose direction of the bone that starts at `joint_name`."""
    children = CHILDREN[joint_name]
    offset = JOINT_BY_NAME[children[0]].offset if children else END_SITES[joint_name]
    return np.array(offset, dtype=float)


def compute_global_rotations(P: np.ndarray) -> Dict[str, R]:
    k = key_positions(P)
    torso_up = k["sh_c"] - k["hip_c"]
    g: Dict[str, R] = {
        "Hips": R.from_matrix(frame_from_x_up(k["L_hip"] - k["R_hip"], torso_up)),
        "Spine": R.from_matrix(frame_from_x_up(k["L_sh"] - k["R_sh"], torso_up)),
        "Head": R.from_matrix(head_frame(k)),
    }
    for joint, (start, end) in BONES.items():
        g[joint] = rotation_between(_rest_direction(joint), k[end] - k[start])
    return g


def compute_local_euler(P: np.ndarray) -> Dict[str, np.ndarray]:
    """Local joint rotations as BVH Euler angles in degrees, channel order Z, X, Y. Each (T, 3)."""
    g = compute_global_rotations(P)
    euler: Dict[str, np.ndarray] = {}
    for name in JOINT_ORDER:
        parent = JOINT_BY_NAME[name].parent
        local = g[name] if parent is None else g[parent].inv() * g[name]
        angles = local.as_euler("ZXY", degrees=True)       # intrinsic ZXY  ==  R = Rz * Rx * Ry
        euler[name] = np.degrees(np.unwrap(np.radians(angles), axis=0))
    return euler
