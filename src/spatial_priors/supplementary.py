"""Numerical methods for the paper's supplementary comparisons.

All response arguments must contain only the selected calibration movies and
repeats. These functions do not choose cohorts or download model/data assets.
Visual methods require aligned frozen-core maps [sample, 512, 16, 24] from the
released core trained on the other animals, plus the exact 64 stimulus anchors.
Location coordinates are feature-grid units, not measured visual degrees.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Sequence

import numpy as np
from scipy import linalg

GRID = (16, 24)
READOUT_ALPHAS = (.01, .1, 1., 10., 100., 1000.)
CELL_CLASSES = frozenset(('L2IT', 'L3IT', 'L4IT', 'L5IT', 'L5ET', 'L5NP', 'L6CT', 'L6IT'))
COARSE_CLASS = {'L2IT': 'L23', 'L3IT': 'L23', 'L4IT': 'L4', 'L5IT': 'L5',
                'L5ET': 'L5', 'L5NP': 'L5', 'L6CT': 'L6', 'L6IT': 'L6'}


def _finite(array, ndim=None):
    if np.iscomplexobj(array):
        raise ValueError('Real-valued inputs are required')
    value = np.asarray(array, dtype=np.float64)
    if (ndim is not None and value.ndim != ndim) or not np.isfinite(value).all():
        raise ValueError('Unexpected dimensions or nonfinite input')
    return value


def normalize_profiles(values):
    """Center each profile, normalize its L2 norm, and return float32 rows.

    Degenerate rows are zero with availability=False, as in the paper.
    """
    value = _finite(values, 2)
    centered = value - value.mean(axis=1, keepdims=True)
    norms = np.sqrt(np.sum(centered * centered, axis=1))
    available = norms >= 1e-8
    output = np.zeros_like(centered)
    output[available] = centered[available] / norms[available, None]
    return output.astype(np.float32), available


def fit_activity_profiles(target, context, sample_ids: Sequence[str], width=64):
    """A1: context-only temporal SVD, with deterministic component orientation."""
    target, context = _finite(target, 2), _finite(context, 2)
    ids = tuple(map(str, sample_ids))
    if len(target) != len(context) or len(ids) != len(target) or len(set(ids)) != len(ids):
        raise ValueError('Target/context samples must have aligned unique IDs')
    if width < 1 or min(context.shape) < width:
        raise ValueError('Insufficient context samples or neurons for the requested rank')

    def standardize(value):
        centered = value - value.mean(axis=0)
        sd = np.sqrt(np.mean(centered * centered, axis=0))
        output = np.zeros_like(centered)
        variable = sd > 0
        output[:, variable] = centered[:, variable] / sd[variable]
        return output

    tz, cz = standardize(target), standardize(context)
    left, singular_values, _ = np.linalg.svd(cz, full_matrices=False)
    basis = left[:, :width].copy()
    for column in range(width):
        absolute = np.abs(basis[:, column])
        tied = np.flatnonzero(absolute == absolute.max())
        pivot = min(tied, key=lambda row: ids[row])
        if basis[pivot, column] < 0:
            basis[:, column] *= -1
    target_coefficients = (basis.T @ tz).T / np.sqrt(len(target))
    context_coefficients = (basis.T @ cz).T / np.sqrt(len(context))
    tp, ta = normalize_profiles(target_coefficients)
    cp, ca = normalize_profiles(context_coefficients)
    return {'target_profiles': tp, 'context_profiles': cp,
            'target_available': ta, 'context_available': ca,
            'basis': basis, 'singular_values': singular_values[:width],
            'target_coefficients': target_coefficients, 'context_coefficients': context_coefficients}


def select_anchor_positions(scan, outer_fold, neuron_indices, unit_identifiers, width=64):
    """Select R2 context anchors by the fixed response-independent hash."""
    if len(neuron_indices) != len(unit_identifiers) or len(neuron_indices) < width:
        raise ValueError('Aligned context identifiers and sufficient anchors are required')
    ranked = []
    for position, (index, unit) in enumerate(zip(neuron_indices, unit_identifiers, strict=True)):
        payload = f'phase8d_R2_anchor_v1|scan={scan}|outer={outer_fold}|unit_id={unit}'
        digest = hashlib.sha256(payload.encode('utf-8')).hexdigest()
        ranked.append((digest, str(unit), int(index), position))
    return np.asarray([row[3] for row in sorted(ranked)[:width]], dtype=np.int64)


def fit_anchor_profiles(target, context, anchor_positions):
    """R2 from [four movies, selected repeats, samples, neurons] arrays.

    The caller selects exactly the budgeted repeats before calling. No split-
    repeat correction or normalization using withheld responses is performed.
    """
    target, context = _finite(target, 4), _finite(context, 4)
    if target.shape[:3] != context.shape[:3] or target.shape[0] != 4:
        raise ValueError('Four aligned calibration movies are required')
    positions = np.asarray(anchor_positions)
    if positions.ndim != 1 or positions.dtype.kind not in 'iu' or len(set(positions)) != len(positions):
        raise ValueError('Anchor positions must be distinct integers')
    if not len(positions) or positions.min() < 0 or positions.max() >= context.shape[-1]:
        raise ValueError('Anchor index outside context population')
    tt = target.mean(axis=1).reshape(-1, target.shape[-1]).T
    ct = context.mean(axis=1).reshape(-1, context.shape[-1]).T
    anchors = ct[positions]

    def correlations(traces):
        a = traces - traces.mean(axis=1, keepdims=True)
        b = anchors - anchors.mean(axis=1, keepdims=True)
        numerator = a @ b.T
        denominator = np.sqrt(np.sum(a*a, axis=1))[:, None] * np.sqrt(np.sum(b*b, axis=1))[None, :]
        result = np.zeros_like(numerator)
        np.divide(numerator, denominator, out=result, where=denominator > 1e-12)
        return np.clip(result, -1, 1)

    tc, cc = correlations(tt), correlations(ct)
    tp, ta = normalize_profiles(tc)
    cp, ca = normalize_profiles(cc)
    return {'target_profiles': tp, 'context_profiles': cp, 'target_available': ta,
            'context_available': ca, 'target_correlations': tc, 'context_correlations': cc}


def profile_pair_features(field, area, xyz, pairs, profiles, activity_profiles=None):
    """Six-feature profile map, or nine-feature visual/activity hybrid map.

    Returns features and a boolean-column mask for the common pair ridge.
    Profiles must already have been centered and L2-normalized.
    """
    xyz, profiles = _finite(xyz, 2), _finite(profiles, 2)
    pairs = _pair_indices(pairs, len(xyz))
    left, right = pairs.T
    field, area = np.asarray(field), np.asarray(area)
    delta = xyz[left] - xyz[right]
    cosine = np.sum(profiles[left] * profiles[right], axis=1)
    columns = [(field[left] == field[right]).astype(float), (area[left] == area[right]).astype(float),
               np.abs(delta[:, 2]), np.sqrt(np.sum(delta*delta, axis=1)), cosine, cosine*cosine]
    if activity_profiles is not None:
        activity = _finite(activity_profiles, 2)
        ac = np.sum(activity[left] * activity[right], axis=1)
        columns.extend([ac, ac*ac, cosine*ac])
    features = np.column_stack(columns)
    return features, np.asarray([True, True] + [False]*(features.shape[1]-2))


def fixed_location_projection():
    """The paper's deterministic signed 512x32 channel projection."""
    signs = np.empty((512, 32), dtype=np.float64)
    for channel in range(512):
        for dimension in range(32):
            payload = f'phase8d_F1_location_projection_v1|channel={channel}|dim={dimension}'
            signs[channel, dimension] = 1 if hashlib.sha256(payload.encode()).digest()[0] & 1 else -1
    return signs / np.sqrt(512.)


def project_core_features(core, cell_chunk=8):
    """Project aligned [sample,512,16,24] core maps to [sample,384,32]."""
    core = np.asarray(core)
    if core.ndim != 4 or core.shape[1:] != (512, *GRID) or cell_chunk < 1:
        raise ValueError('Expected [sample,512,16,24] and positive chunk size')
    flat = core.reshape(len(core), 512, 384)
    projection = fixed_location_projection()
    output = np.empty((len(core), 384, 32), dtype=np.float64)
    for start in range(0, 384, cell_chunk):
        stop = min(start + cell_chunk, 384)
        block = _finite(flat[:, :, start:stop])
        matrix = np.ascontiguousarray(block.transpose(0, 2, 1)).reshape(-1, 512)
        output[:, start:stop] = (matrix @ projection).reshape(len(core), stop-start, 32)
    return output


def _movie_partition(movie_ids, n_samples, movie_order=None):
    movies = np.asarray(tuple(map(str, movie_ids)))
    order = tuple(dict.fromkeys(movies)) if movie_order is None else tuple(map(str, movie_order))
    if movies.shape != (n_samples,) or len(order) != 4 or len(set(order)) != 4 or set(movies) != set(order):
        raise ValueError('Exactly four training movies must partition all samples')
    if any(np.sum(movies == movie) < 2 for movie in order):
        raise ValueError('Each movie needs at least two samples')
    return movies, order


def _pearson(y, predicted):
    """Batch Pearson: predicted[location,sample,neuron], invalid scores -1."""
    yc = y - y.mean(axis=0)
    pc = predicted - predicted.mean(axis=1, keepdims=True)
    numerator = np.einsum('tn,ktn->kn', yc, pc, optimize=True)
    yn = np.sqrt(np.sum(yc*yc, axis=0))
    pn = np.sqrt(np.sum(pc*pc, axis=1))
    denominator = pn*yn[None, :]
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (pn >= 1e-12) & (yn[None, :] >= 1e-12)
    scores = np.full(numerator.shape, -1., dtype=np.float64)
    scores[valid] = np.clip(numerator[valid]/denominator[valid], -1., 1.)
    return scores, valid


def fit_locations(projected, responses, movie_ids, movie_order=None, cell_chunk=8):
    """Four-fold ridge location search and its overlapping-fold stability check.

    No neurons are silently removed. ``usable`` reports all-fold validity and
    nonflat scores; weak but finite fits stay in the cohort. The caller must
    check this flag before using the complete descriptor regime.
    """
    x, y = _finite(projected, 3), _finite(responses, 2)
    if x.shape[1:] != (384, 32) or len(x) != len(y) or y.shape[1] < 1 or cell_chunk < 1:
        raise ValueError('Expected projected[sample,384,32] and aligned responses')
    movies, order = _movie_partition(movie_ids, len(y), movie_order)
    scores = np.empty((4, 384, y.shape[1]))
    valid = np.empty(scores.shape, dtype=bool)
    for fold, movie in enumerate(order):
        validation = movies == movie
        training = ~validation
        train_y = y[training]
        mean_y = train_y.mean(axis=0)
        centered_y = train_y - mean_y
        for start in range(0, 384, cell_chunk):
            stop = min(start+cell_chunk, 384)
            count = stop-start
            raw = x[training, start:stop]
            mean, sd = raw.mean(axis=0), raw.std(axis=0, ddof=0)
            scale = np.where(sd < 1e-8, 1., sd)
            standardized = (raw-mean)/scale
            design = standardized.transpose(1, 0, 2)
            gram = design.transpose(0, 2, 1) @ design + np.eye(32)[None, :, :]
            cross = (np.ascontiguousarray(standardized).reshape(len(raw), count*32).T @ centered_y).reshape(count, 32, y.shape[1])
            weights = np.empty_like(cross)
            for cell in range(count):
                factor = linalg.cho_factor(gram[cell], lower=True, check_finite=False)
                weights[cell] = linalg.cho_solve(factor, cross[cell], check_finite=False)
            prediction = ((x[validation, start:stop]-mean)/scale).transpose(1, 0, 2) @ weights + mean_y
            scores[fold, start:stop], valid[fold, start:stop] = _pearson(y[validation], prediction)
    means = np.round(scores.mean(axis=0), 12)
    chosen = np.argmax(means, axis=0)
    neuron = np.arange(y.shape[1])
    selected_valid = valid[:, chosen, neuron].T
    ties = np.sum(means == means[chosen, neuron][None, :], axis=0)
    row, column = chosen//24, chosen % 24
    leave_one_out = np.stack([np.argmax(np.round(np.delete(scores, fold, axis=0).mean(axis=0), 12), axis=0)
                             for fold in range(4)], axis=1)
    dispersion = np.sqrt(np.mean((leave_one_out//24-row[:, None])**2 + (leave_one_out % 24-column[:, None])**2, axis=1))
    return {'selected_cell': chosen, 'selected_row': row, 'selected_column': column,
            'selected_mean_score': means[chosen, neuron], 'selected_fold_scores': scores[:, chosen, neuron].T,
            'usable': selected_valid.all(axis=1) & (ties < 384), 'selected_fold_validity': selected_valid,
            'location_fold_scores': scores, 'location_mean_scores': means,
            'leave_one_fold_out_cells': leave_one_out, 'leave_one_fold_out_rms': dispersion}


def location_pair_distances(locations, pairs):
    """Euclidean pair distance in equally weighted feature row/column units."""
    points = _finite(locations, 2)
    if points.shape[1] != 2:
        raise ValueError('Locations must have two columns: row and column')
    left, right = _pair_indices(pairs, len(points)).T
    return np.sqrt(np.sum((points[left]-points[right])**2, axis=1))


def _ridge_weights(design, centered_y, alpha):
    if design.shape[0] < design.shape[1]:
        gram = design @ design.T + alpha*np.eye(len(design))
        return design.T @ linalg.cho_solve(linalg.cho_factor(gram, lower=True), centered_y)
    gram = design.T @ design + alpha*np.eye(design.shape[1])
    return linalg.cho_solve(linalg.cho_factor(gram, lower=True), design.T @ centered_y)


def fit_stimulus_profiles(features, responses, movie_ids, anchor_features, movie_order=None):
    """Readout ridge with LOMO alpha selection; valid for S1 or one F1 location.

    Use S1's 248 aligned handcrafted features, or the 512 channels at each
    recording's chosen location. Anchors must be the paper's 64 training-only
    stimulus identities. This routine fits readouts, not the frozen visual core.
    """
    x, y, anchors = _finite(features, 2), _finite(responses, 2), _finite(anchor_features, 2)
    if len(x) != len(y) or anchors.shape != (64, x.shape[1]):
        raise ValueError('Aligned samples and exactly 64 compatible anchors are required')
    movies, order = _movie_partition(movie_ids, len(x), movie_order)
    scores = np.empty((y.shape[1], len(READOUT_ALPHAS), 4))
    for fold, movie in enumerate(order):
        validation = movies == movie
        mean, sd = x[~validation].mean(axis=0), x[~validation].std(axis=0, ddof=0)
        scale = np.where(sd < 1e-8, 1., sd)
        train = (x[~validation]-mean)/scale
        test = (x[validation]-mean)/scale
        ym = y[~validation].mean(axis=0)
        for index, alpha in enumerate(READOUT_ALPHAS):
            prediction = test @ _ridge_weights(train, y[~validation]-ym, alpha) + ym
            scores[:, index, fold] = _pearson(y[validation], prediction[None, :, :])[0][0]
    rounded = np.round(scores.mean(axis=2), 12)
    selected = len(READOUT_ALPHAS)-1-np.argmax(rounded[:, ::-1], axis=1)
    mean, sd = x.mean(axis=0), x.std(axis=0, ddof=0)
    scale = np.where(sd < 1e-8, 1., sd)
    design = (x-mean)/scale
    ym = y.mean(axis=0)
    weights = np.empty((y.shape[1], x.shape[1]))
    for index, alpha in enumerate(READOUT_ALPHAS):
        mask = selected == index
        if mask.any():
            weights[mask] = _ridge_weights(design, (y-ym)[:, mask], alpha).T
    predictions = (((anchors-mean)/scale) @ weights.T + ym).T
    profiles, available = normalize_profiles(predictions)
    return {'profiles': profiles, 'available': available, 'anchor_predictions': predictions,
            'selected_alpha': np.asarray(READOUT_ALPHAS)[selected], 'weights': weights,
            'bias': ym, 'feature_mean': mean, 'feature_scale': scale, 'alpha_fold_scores': scores}


def fit_visual_profiles(core, responses, movie_ids, anchor_core, movie_order=None):
    """F1: fit fresh locations and full-channel readouts on budgeted samples."""
    core, y, anchors = np.asarray(core), _finite(responses, 2), np.asarray(anchor_core)
    if anchors.shape != (64, 512, *GRID) or len(core) != len(y):
        raise ValueError('Expected aligned core and 64 matching anchor core maps')
    locations = fit_locations(project_core_features(core), y, movie_ids, movie_order)
    profiles = np.empty((y.shape[1], 64), dtype=np.float32)
    alphas = np.empty(y.shape[1])
    available = np.empty(y.shape[1], dtype=bool)
    for cell in np.unique(locations['selected_cell']):
        row, column = divmod(int(cell), 24)
        mask = locations['selected_cell'] == cell
        fitted = fit_stimulus_profiles(core[:, :, row, column], y[:, mask], movie_ids,
                                       anchors[:, :, row, column], movie_order)
        profiles[mask], alphas[mask], available[mask] = fitted['profiles'], fitted['selected_alpha'], fitted['available']
    return {'profiles': profiles, 'available': available, 'selected_alpha': alphas, 'locations': locations}


def _pair_indices(pairs, size):
    pairs = np.asarray(pairs)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in 'iu':
        raise ValueError('Pairs must have integer shape [pair,2]')
    if pairs.size and (pairs.min() < 0 or pairs.max() >= size):
        raise ValueError('Pair index outside supplied recording population')
    return pairs.astype(np.int64, copy=False)


@dataclass
class ClassPairEncoder:
    """Training-pair vocabulary, with explicit missing/unseen output slots."""
    labels: tuple
    vocabulary: tuple
    missing_supported: bool

    @classmethod
    def fit(cls, labels, training_pairs, coarse=False):
        labels = tuple((COARSE_CLASS.get(value) if coarse else value)
                       if value in CELL_CLASSES else None for value in labels)
        pairs = _pair_indices(training_pairs, len(labels))
        known, missing = set(), False
        for first, second in pairs:
            a, b = labels[first], labels[second]
            if a is None or b is None:
                missing = True
            else:
                known.add(tuple(sorted((a, b))))
        return cls(labels, tuple(sorted(known)), missing)

    def transform(self, pairs):
        pairs = _pair_indices(pairs, len(self.labels))
        vocabulary = {pair: index for index, pair in enumerate(self.vocabulary)}
        output = np.zeros((len(pairs), len(vocabulary)+2))
        for row, (first, second) in enumerate(pairs):
            a, b = self.labels[first], self.labels[second]
            if a is None or b is None:
                if self.missing_supported:
                    output[row, -2] = 1
            else:
                column = vocabulary.get(tuple(sorted((a, b))))
                if column is not None:
                    output[row, column] = 1
        return output


def shared_input_features(source_counts, endpoint_known, pairs):
    """Unscaled Q7/H1 from integer counts with axes [recording, source].

    Unknown endpoints and zero-norm undefined overlap remain NaN, distinct from
    defined zero overlap. Biological-identity exclusions are applied upstream.
    """
    counts = np.asarray(source_counts)
    known = np.asarray(endpoint_known, dtype=bool)
    if counts.ndim != 2 or counts.dtype.kind not in 'iu' or np.any(counts < 0):
        raise ValueError('Nonnegative integer synapse counts are required')
    if known.shape != (len(counts),):
        raise ValueError('One eligibility flag per endpoint is required')
    totals = np.asarray([sum(map(int, row)) for row in counts], dtype=np.int64)
    if max(map(int, totals), default=0)**2 > np.iinfo(np.int64).max:
        raise OverflowError('Exact synapse inner products could exceed int64')
    counts = counts.astype(np.int64, copy=False)
    left, right = _pair_indices(pairs, len(counts)).T
    both_known = known[left] & known[right]
    synapses = np.log1p(totals.astype(np.float64))
    sources = np.log1p((counts > 0).sum(axis=1))
    squared_norms = np.sum(counts*counts, axis=1).astype(np.float64)
    denominator = np.sqrt(squared_norms[left]*squared_norms[right])
    defined = both_known & (denominator > 0)
    overlap = np.full(len(left), np.nan)
    overlap[defined] = np.sum(counts[left[defined]]*counts[right[defined]], axis=1)/denominator[defined]
    q = np.column_stack([synapses[left]+synapses[right], np.abs(synapses[left]-synapses[right]),
                         sources[left]+sources[right], np.abs(sources[left]-sources[right]),
                         both_known, both_known, defined]).astype(float)
    q[~both_known, :4] = np.nan
    return q, overlap[:, None]


@dataclass
class OptionalFeatureScaler:
    """Train-pair scaling for anatomy/graph features with missing support.

    Missing, unsupported, and exactly constant columns contribute zero. Varying
    binary columns remain unscaled. Do not standardize this output again.
    """
    means: np.ndarray
    scales: np.ndarray
    enabled: np.ndarray
    binary: np.ndarray

    @classmethod
    def fit(cls, features, binary=None):
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or np.isinf(x).any():
            raise ValueError('Expected two-dimensional finite/NaN features')
        binary = np.zeros(x.shape[1], dtype=bool) if binary is None else np.asarray(binary, dtype=bool)
        if binary.shape != (x.shape[1],):
            raise ValueError('Binary mask width differs')
        means, scales, enabled = np.zeros(x.shape[1]), np.ones(x.shape[1]), np.zeros(x.shape[1], dtype=bool)
        for column in range(x.shape[1]):
            observed = x[np.isfinite(x[:, column]), column]
            if not len(observed):
                continue
            means[column] = math.fsum(map(float, observed))/len(observed)
            enabled[column] = np.any(observed != observed[0])
            sd = math.sqrt(math.fsum((float(v)-means[column])**2 for v in observed)/len(observed))
            scales[column] = sd if sd >= 1e-8 else 1.
        return cls(means, scales, enabled, binary)

    def transform(self, features):
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.means) or np.isinf(x).any():
            raise ValueError('Invalid optional feature matrix')
        output = np.zeros_like(x)
        for column in np.flatnonzero(self.enabled):
            mask = np.isfinite(x[:, column])
            output[mask, column] = (x[mask, column] if self.binary[column]
                                    else (x[mask, column]-self.means[column])/self.scales[column])
        return output
