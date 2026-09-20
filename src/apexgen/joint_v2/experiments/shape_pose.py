"""A rigid-fit definition of separate shape and pose interventions."""
import numpy as np


def rigid_map(mobile, reference):
    a, b = np.asarray(mobile, dtype=float), np.asarray(reference, dtype=float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[-1] != 3 or len(a) < 3:
        raise ValueError('Expected matched [L>=3,3] coordinates')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite coordinates')
    u, _, vh = np.linalg.svd((a-a.mean(0)).T @ (b-b.mean(0)))
    sign = np.diag([1., 1., np.linalg.det(u @ vh)])
    r = u @ sign @ vh
    return r, b.mean(0)-a.mean(0) @ r


def factor_shape_pose(baseline, revised):
    """Four views; original pose is defined by a CA least-squares alignment."""
    b, q = np.asarray(baseline, dtype=float), np.asarray(revised, dtype=float)
    if b.shape != q.shape or b.ndim != 3 or b.shape[1:] != (3, 3):
        raise ValueError('Expected matching N/CA/C [L,3,3] backbones')
    r, t = rigid_map(b[:, 1], q[:, 1])
    return dict(baseline=b.copy(), pose_only=b @ r+t,
                shape_only=(q-t) @ r.T, shape_pose=q.copy())
