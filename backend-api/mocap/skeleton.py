"""The target humanoid skeleton (T-pose rig) that the captured motion is applied to.

Units: centimetres. Axes: +Y up, +X = character's left, +Z = character's front.
In the rest (T-)pose every joint's global rotation is identity, so a bone's rest
direction is simply its child's OFFSET.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Joint:
    name: str
    parent: Optional[str]
    offset: Tuple[float, float, float]


# Order matters: children are written in this order, parents always come first.
JOINTS: List[Joint] = [
    Joint("Hips", None, (0, 0, 0)),                 # root: position channels carry the translation
    Joint("Spine", "Hips", (0, 50, 0)),             # sits at the mid-shoulder point
    Joint("Head", "Spine", (0, 25, 0)),
    Joint("LeftArm", "Spine", (18, 0, 0)),          # shoulder joint
    Joint("LeftForeArm", "LeftArm", (28, 0, 0)),    # elbow
    Joint("LeftHand", "LeftForeArm", (26, 0, 0)),   # wrist
    Joint("RightArm", "Spine", (-18, 0, 0)),
    Joint("RightForeArm", "RightArm", (-28, 0, 0)),
    Joint("RightHand", "RightForeArm", (-26, 0, 0)),
    Joint("LeftUpLeg", "Hips", (9, 0, 0)),          # hip joint
    Joint("LeftLeg", "LeftUpLeg", (0, -43, 0)),     # knee
    Joint("LeftFoot", "LeftLeg", (0, -40, 0)),      # ankle
    Joint("RightUpLeg", "Hips", (-9, 0, 0)),
    Joint("RightLeg", "RightUpLeg", (0, -43, 0)),
    Joint("RightFoot", "RightLeg", (0, -40, 0)),
]

# Leaf bones need an "End Site" so BVH viewers know where the bone ends.
END_SITES: Dict[str, Tuple[float, float, float]] = {
    "Head": (0, 20, 0),
    "LeftHand": (8, 0, 0),
    "RightHand": (-8, 0, 0),
    "LeftFoot": (0, -6, 14),
    "RightFoot": (0, -6, 14),
}

JOINT_BY_NAME: Dict[str, Joint] = {j.name: j for j in JOINTS}
CHILDREN: Dict[str, List[str]] = {j.name: [] for j in JOINTS}
for _j in JOINTS:
    if _j.parent is not None:
        CHILDREN[_j.parent].append(_j.name)


def _dfs(name: str, out: List[str]) -> None:
    out.append(name)
    for child in CHILDREN[name]:
        _dfs(child, out)


# Pre-order traversal = the order in which BVH channel values appear in every frame.
JOINT_ORDER: List[str] = []
_dfs("Hips", JOINT_ORDER)
