"""Full processing pipeline: raw webcam landmarks -> cleaned data -> joint rotations -> .bvh file."""
from pathlib import Path
from typing import Any, Dict

import numpy as np

from . import landmarks as L
from .bvh_export import write_bvh
from .landmarks import mediapipe_to_yup
from .rotations import compute_local_euler
from .smoothing import despike, fill_low_visibility, kalman_smooth, resample_uniform

MIN_FRAMES = 15
HIP_HEIGHT_CM = 100.0       # rest height of the Hips joint above the floor
SHOULDER_WIDTH_M = 0.38     # used to turn image movement into metres


def estimate_root_translation(image: np.ndarray, aspect: float, fps: float) -> np.ndarray:
    """Approximate where the person moves inside the frame. image: (T, 33, >=2) normalised x, y coords.

    MediaPipe world landmarks are centred on the hips, so they contain no travel. We recover
    left/right and up/down movement from the image-space hip position (scaled using shoulder
    width). Depth (towards/away from camera) is NOT recovered: Z stays 0.
    """
    xy_scale = np.array([aspect, 1.0])
    shoulders = np.linalg.norm((image[:, L.L_SHOULDER, :2] - image[:, L.R_SHOULDER, :2]) * xy_scale, axis=1)
    metres_per_unit = SHOULDER_WIDTH_M / max(float(np.median(shoulders)), 1e-6)
    hip = (image[:, L.L_HIP, :2] + image[:, L.R_HIP, :2]) / 2
    disp = (hip - hip[0]) * xy_scale * metres_per_unit
    root = np.zeros((image.shape[0], 3))
    root[:, 0] = disp[:, 0] * 100.0                        # image right  -> +X
    root[:, 1] = HIP_HEIGHT_CM - disp[:, 1] * 100.0        # image down   -> -Y
    return kalman_smooth(root[:, None, :], fps)[:, 0, :]


def process_motion(doc: Dict[str, Any], bvh_path: Path, target_fps: int = 30) -> Dict[str, Any]:
    times = np.asarray(doc["timestamps"], dtype=float)
    world = np.asarray(doc["world"], dtype=float)        # (T, 33, 4): x, y, z, visibility
    image = np.asarray(doc["image"], dtype=float)        # (T, 33, 4)
    if len(times) < MIN_FRAMES:
        raise ValueError(f"Need at least {MIN_FRAMES} frames, got {len(times)}")

    # 1. low-visibility points -> interpolate
    world_xyz = fill_low_visibility(world[..., :3], world[..., 3])
    image_xy = fill_low_visibility(image[..., :3], image[..., 3])

    # 2. constant frame rate
    new_times, world_xyz = resample_uniform(times, world_xyz, target_fps)
    _, image_xy = resample_uniform(times, image_xy, target_fps)
    if len(new_times) < MIN_FRAMES:
        raise ValueError("Recording is too short")

    # 3. remove glitches, then smooth
    world_xyz = despike(world_xyz)
    world_xyz = kalman_smooth(world_xyz, target_fps)

    # 4. MediaPipe axes -> Y-up skeleton axes, then joint rotations
    euler = compute_local_euler(mediapipe_to_yup(world_xyz))

    # 5. root movement and export
    aspect = float(doc["video_width"]) / float(doc["video_height"])
    root = estimate_root_translation(image_xy, aspect, target_fps)
    write_bvh(bvh_path, euler, root, target_fps)

    return {"frames": int(len(new_times)), "fps": target_fps, "duration_s": round(len(new_times) / target_fps, 2)}
