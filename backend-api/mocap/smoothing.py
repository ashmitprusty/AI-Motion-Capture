"""Data cleaning: gap filling, spike removal, resampling and Kalman smoothing."""
import numpy as np
from scipy.ndimage import median_filter


def fill_low_visibility(xyz: np.ndarray, visibility: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Mark landmarks with low visibility as missing, then linearly interpolate over time.

    xyz: (T, N, 3), visibility: (T, N).
    """
    out = xyz.astype(float).copy()
    out[visibility < threshold] = np.nan
    idx = np.arange(out.shape[0])
    for n in range(out.shape[1]):
        for d in range(out.shape[2]):
            col = out[:, n, d]  # a view into `out`
            bad = np.isnan(col)
            if bad.all():
                col[:] = 0.0
            elif bad.any():
                col[bad] = np.interp(idx[bad], idx[~bad], col[~bad])
    return out


def resample_uniform(times: np.ndarray, data: np.ndarray, fps: float):
    """Resample irregularly timed frames onto a fixed frame rate (linear interpolation).

    times: (T,) seconds, data: (T, ...). Returns (new_times, new_data).
    """
    order = np.argsort(times)
    times, data = times[order], data[order]
    times, unique_idx = np.unique(times, return_index=True)
    data = data[unique_idx]
    times = times - times[0]
    new_times = np.arange(0.0, times[-1], 1.0 / fps)
    flat = data.reshape(len(times), -1)
    resampled = np.stack([np.interp(new_times, times, flat[:, i]) for i in range(flat.shape[1])], axis=1)
    return new_times, resampled.reshape((len(new_times),) + data.shape[1:])


def despike(points: np.ndarray, window: int = 5, n_sigmas: float = 3.0, floor: float = 0.01) -> np.ndarray:
    """Hampel filter: replace sudden one-frame jumps with the local median.

    points: (T, N, 3) in metres. `floor` (1 cm) stops it from touching normal jitter.
    """
    med = median_filter(points, size=(window, 1, 1), mode="nearest")
    diff = np.abs(points - med)
    mad = median_filter(diff, size=(window, 1, 1), mode="nearest")
    limit = n_sigmas * 1.4826 * mad + floor
    out = points.copy()
    mask = diff > limit
    out[mask] = med[mask]
    return out


def _kalman_pass(z: np.ndarray, dt: float, process_noise: float, measurement_noise: float) -> np.ndarray:
    """Constant-velocity Kalman filter run over all series at once. z: (T, S)."""
    T, S = z.shape
    F = np.array([[1.0, dt], [0.0, 1.0]])
    Q = process_noise * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])
    x = np.zeros((2, S))
    x[0] = z[0]
    P = np.eye(2)
    out = np.empty_like(z)
    out[0] = z[0]
    for t in range(1, T):
        x = F @ x                      # predict
        P = F @ P @ F.T + Q
        K = P[:, 0] / (P[0, 0] + measurement_noise)
        x = x + K[:, None] * (z[t] - x[0])[None, :]   # update
        P = P - np.outer(K, P[0, :])
        out[t] = x[0]
    return out


def kalman_smooth(points: np.ndarray, fps: float, process_noise: float = 50.0,
                  measurement_noise: float = 0.02 ** 2) -> np.ndarray:
    """Zero-lag Kalman smoothing (forward pass + backward pass, averaged).

    points: (T, N, 3) in metres. Raise `measurement_noise` for smoother (but softer) motion.
    """
    T = points.shape[0]
    if T < 3:
        return points.copy()
    dt = 1.0 / fps
    z = points.reshape(T, -1)
    fwd = _kalman_pass(z, dt, process_noise, measurement_noise)
    bwd = _kalman_pass(z[::-1], dt, process_noise, measurement_noise)[::-1]
    return (0.5 * (fwd + bwd)).reshape(points.shape)
