"""Write the animation as a BVH file (readable by Blender, Unity, Unreal, MotionBuilder, three.js)."""
from pathlib import Path
from typing import Dict

import numpy as np

from .skeleton import CHILDREN, END_SITES, JOINT_BY_NAME, JOINT_ORDER


def _fmt(v) -> str:
    return " ".join(f"{x:.6f}" for x in v)


def _hierarchy_lines(name: str, depth: int, lines: list) -> None:
    pad = "  " * depth
    joint = JOINT_BY_NAME[name]
    lines.append(f"{pad}{'ROOT' if joint.parent is None else 'JOINT'} {name}")
    lines.append(f"{pad}{{")
    lines.append(f"{pad}  OFFSET {_fmt(joint.offset)}")
    if joint.parent is None:
        lines.append(f"{pad}  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation")
    else:
        lines.append(f"{pad}  CHANNELS 3 Zrotation Xrotation Yrotation")
    for child in CHILDREN[name]:
        _hierarchy_lines(child, depth + 1, lines)
    if name in END_SITES:
        lines.append(f"{pad}  End Site")
        lines.append(f"{pad}  {{")
        lines.append(f"{pad}    OFFSET {_fmt(END_SITES[name])}")
        lines.append(f"{pad}  }}")
    lines.append(f"{pad}}}")


def write_bvh(path: Path, euler: Dict[str, np.ndarray], root_position_cm: np.ndarray, fps: float) -> Path:
    """euler[name]: (T, 3) degrees (Z, X, Y). root_position_cm: (T, 3)."""
    n_frames = root_position_cm.shape[0]
    lines = ["HIERARCHY"]
    _hierarchy_lines("Hips", 0, lines)
    lines += ["MOTION", f"Frames: {n_frames}", f"Frame Time: {1.0 / fps:.6f}"]
    for t in range(n_frames):
        values = list(root_position_cm[t])
        for name in JOINT_ORDER:
            values += list(euler[name][t])
        lines.append(" ".join(f"{v:.5f}" for v in values))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
