"""MediaPipe Pose landmark indices and coordinate-system conversion."""
import numpy as np

NUM_LANDMARKS = 33

NOSE = 0
L_EAR, R_EAR = 7, 8
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_PINKY, R_PINKY = 17, 18
L_INDEX, R_INDEX = 19, 20
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_FOOT_INDEX, R_FOOT_INDEX = 31, 32


def mediapipe_to_yup(points: np.ndarray) -> np.ndarray:
    """Convert MediaPipe world coordinates to a right-handed, Y-up system.

    MediaPipe (unmirrored camera frame): +x = image right, +y = DOWN,
    +z = away from the camera.
    Output (and BVH skeleton) system:     +x = person's LEFT, +y = UP,
    +z = the direction the person faces (towards the camera).

    This is a proper rotation (180 degrees about X), so no mirroring occurs.
    """
    out = np.array(points, dtype=float, copy=True)
    out[..., 1] *= -1.0
    out[..., 2] *= -1.0
    return out
