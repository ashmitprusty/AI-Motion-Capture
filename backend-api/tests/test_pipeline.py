"""Offline test of the whole backend pipeline using a synthetic 'person' (no camera needed).

Run:  python tests/test_pipeline.py      (or: pytest)

It builds a fake MediaPipe recording of a person raising/bending their left arm and walking
sideways, runs the pipeline, then re-computes forward kinematics from the BVH file and checks
that the exported bones point in the same directions as the input landmarks.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mocap import landmarks as L                      # noqa: E402
from mocap.pipeline import process_motion             # noqa: E402
from mocap.skeleton import JOINT_BY_NAME, JOINT_ORDER  # noqa: E402


def make_pose_yup(t: float) -> np.ndarray:
    """(33, 3) Y-up pose in metres: T-pose with the left arm swinging."""
    P = np.zeros((33, 3))
    P[L.L_HIP], P[L.R_HIP] = (0.09, 0, 0), (-0.09, 0, 0)
    P[L.L_SHOULDER], P[L.R_SHOULDER] = (0.18, 0.5, 0), (-0.18, 0.5, 0)
    P[L.L_EAR], P[L.R_EAR], P[L.NOSE] = (0.08, 0.72, 0), (-0.08, 0.72, 0), (0, 0.70, 0.10)
    a = np.radians(40 * np.sin(2 * np.pi * 0.5 * t))            # upper arm angle in the XY plane
    b = np.radians(50 + 30 * np.sin(2 * np.pi * 0.8 * t))       # extra elbow bend
    d1 = np.array([np.cos(a), np.sin(a), 0.0])
    d2 = np.array([np.cos(a + b), np.sin(a + b), 0.0])
    P[L.L_ELBOW] = P[L.L_SHOULDER] + 0.28 * d1
    P[L.L_WRIST] = P[L.L_ELBOW] + 0.26 * d2
    P[L.L_INDEX] = P[L.L_WRIST] + 0.08 * d2 + (0, 0, 0.01)
    P[L.L_PINKY] = P[L.L_WRIST] + 0.08 * d2 - (0, 0, 0.01)
    P[L.R_ELBOW], P[L.R_WRIST] = (-0.46, 0.5, 0), (-0.72, 0.5, 0)
    P[L.R_INDEX], P[L.R_PINKY] = (-0.80, 0.5, 0.01), (-0.80, 0.5, -0.01)
    for side, sx in (("L", 1), ("R", -1)):
        P[getattr(L, f"{side}_KNEE")] = (0.09 * sx, -0.43, 0)
        P[getattr(L, f"{side}_ANKLE")] = (0.09 * sx, -0.83, 0)
        P[getattr(L, f"{side}_FOOT_INDEX")] = (0.09 * sx, -0.89, 0.14)
    return P


def make_doc(seconds=4.0, cam_fps=27, seed=1):
    rng = np.random.default_rng(seed)
    ts = np.cumsum(rng.uniform(0.8, 1.2, int(seconds * cam_fps)) / cam_fps)   # irregular timing
    world, image = [], []
    for t in ts:
        yup = make_pose_yup(t)
        mp = yup.copy(); mp[:, 1] *= -1; mp[:, 2] *= -1                        # -> MediaPipe axes
        mp = mp + rng.normal(0, 0.004, mp.shape)                               # sensor noise
        world.append(np.hstack([mp, np.ones((33, 1))]).tolist())
        img = np.zeros((33, 4)); img[:, 3] = 1
        cx = 0.4 + 0.05 * t                                                     # walks to the right
        img[L.L_HIP, :2], img[L.R_HIP, :2] = (cx + 0.03, 0.6), (cx - 0.03, 0.6)
        img[L.L_SHOULDER, :2], img[L.R_SHOULDER, :2] = (cx + 0.06, 0.4), (cx - 0.06, 0.4)
        image.append(img.tolist())
    # inject a one-frame glitch and a short dropout to prove cleaning works
    world[20][L.L_ELBOW][0] += 0.8
    for f in range(40, 44):
        world[f][L.L_WRIST][3] = 0.1
    return {"timestamps": ts.tolist(), "world": world, "image": image, "video_width": 640, "video_height": 480}


def parse_bvh_motion(path: Path):
    lines = path.read_text().splitlines()
    i = lines.index("MOTION")
    n_frames = int(lines[i + 1].split(":")[1])
    frame_time = float(lines[i + 2].split(":")[1])
    data = np.array([[float(v) for v in ln.split()] for ln in lines[i + 3:i + 3 + n_frames]])
    return n_frames, frame_time, data


def forward_kinematics(frame_values: np.ndarray):
    """Global joint rotations & positions (cm) from one BVH frame, using the same rules as BVH."""
    root = frame_values[:3]
    rots = {name: R.from_euler("ZXY", frame_values[3 + 3 * k:6 + 3 * k], degrees=True)
            for k, name in enumerate(JOINT_ORDER)}
    g_rot, pos = {}, {}
    for name in JOINT_ORDER:
        j = JOINT_BY_NAME[name]
        if j.parent is None:
            g_rot[name], pos[name] = rots[name], root.copy()
        else:
            g_rot[name] = g_rot[j.parent] * rots[name]
            pos[name] = pos[j.parent] + g_rot[j.parent].apply(np.array(j.offset, float))
    return g_rot, pos


def angle_deg(u, v):
    u, v = u / np.linalg.norm(u), v / np.linalg.norm(v)
    return np.degrees(np.arccos(np.clip(u @ v, -1, 1)))


def test_pipeline_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "test.bvh"
        doc = make_doc()
        t0 = doc["timestamps"][0]          # the pipeline re-bases time so the clip starts at 0
        stats = process_motion(doc, out, target_fps=30)
        assert out.exists() and out.read_text().startswith("HIERARCHY")
        n_frames, frame_time, data = parse_bvh_motion(out)
        assert n_frames == stats["frames"] > 90
        assert abs(frame_time - 1 / 30) < 1e-4
        assert data.shape[1] == 3 + 3 * len(JOINT_ORDER)
        assert np.isfinite(data).all()

        # Root travelled to the right (+X) as the synthetic person walked.
        assert data[-1, 0] > data[0, 0] + 5

        worst = 0.0
        for fi in range(10, n_frames - 10, 7):     # skip the edges
            t = t0 + fi / 30.0
            P = make_pose_yup(t)
            g_rot, pos = forward_kinematics(data[fi])
            for joint, a_idx, b_idx in [("LeftArm", L.L_SHOULDER, L.L_ELBOW),
                                        ("LeftForeArm", L.L_ELBOW, L.L_WRIST),
                                        ("LeftUpLeg", L.L_HIP, L.L_KNEE),
                                        ("RightArm", L.R_SHOULDER, L.R_ELBOW)]:
                child = [c for c in JOINT_ORDER if JOINT_BY_NAME[c].parent == joint][0]
                bone = pos[child] - pos[joint]
                worst = max(worst, angle_deg(bone, P[b_idx] - P[a_idx]))
        print(f"worst bone direction error: {worst:.2f} degrees")
        assert worst < 8.0


if __name__ == "__main__":
    test_pipeline_end_to_end()
    print("PASS")
