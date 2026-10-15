"""Context-only low-rank projection used by the fixed reference learning curve."""
from __future__ import annotations

import numpy as np

RANKS = ('8', '16', '32', '64', '128', 'full')


def correlation_family(target, context, context_ids, ranks=RANKS):
    """Project calibration correlations onto right singular vectors of context.

    The full-rank member is the exact empirical Pearson matrix. Numeric ranks
    are never silently capped; data must support every requested rank.
    """
    target = np.asarray(target, dtype=np.float64)
    context = np.asarray(context, dtype=np.float64)
    if target.ndim != 2 or context.ndim != 2 or len(target) != len(context) or len(target) < 3:
        raise ValueError('Target/context must be aligned matrices with at least 3 samples')
    if not np.isfinite(target).all() or not np.isfinite(context).all():
        raise ValueError('Calibration data must be finite')
    if len(context_ids) != context.shape[1] or len(set(context_ids)) != len(context_ids):
        raise ValueError('Context identifiers must be unique and match the columns')
    maximum = min(len(context) - 1, context.shape[1])
    if not ranks or len(set(ranks)) != len(ranks):
        raise ValueError('Rank labels must be unique and nonempty')
    for rank in ranks:
        if rank not in RANKS or (rank != 'full' and int(rank) > maximum):
            raise ValueError(f'Unsupported rank {rank}; available rank is {maximum}')
    means = context.mean(axis=0, dtype=np.float64)
    centered = context - means[None, :]
    scales = np.sqrt(np.mean(centered * centered, axis=0, dtype=np.float64))
    variable = scales >= 1e-8
    scales[~variable] = 1.0
    standardized = centered / scales[None, :]
    standardized[:, ~variable] = 0.0
    target_centered = target - target.mean(axis=0, keepdims=True)
    context_centered = context - context.mean(axis=0, keepdims=True)
    numerator = target_centered.T @ context_centered
    denominator = np.sqrt(np.sum(target_centered ** 2, axis=0))[:, None] * np.sqrt(np.sum(context_centered ** 2, axis=0))[None, :]
    available = np.isfinite(denominator) & (denominator > 1e-12)
    raw = np.zeros_like(numerator)
    raw[available] = numerator[available] / denominator[available]
    raw = np.clip(raw, -1.0, 1.0)
    _, _, right = np.linalg.svd(standardized, full_matrices=False)
    right = right[:maximum].copy()
    for row in range(len(right)):
        magnitude = np.abs(right[row])
        tied = np.flatnonzero(np.isclose(magnitude, float(magnitude.max()), rtol=0.0, atol=1e-15))
        pivot = min((int(index) for index in tied), key=lambda index: (context_ids[index], index))
        if right[row, pivot] < 0.0:
            right[row] *= -1.0
    result = {}
    for rank in ranks:
        if rank == 'full':
            result[rank] = raw.copy()
        else:
            basis = right[:int(rank)]
            shrunk = (raw @ basis.T) @ basis
            shrunk[:, ~variable] = 0.0
            shrunk[~np.any(available, axis=1), :] = 0.0
            result[rank] = shrunk
    return result
