"""Selection, prediction, scoring and aggregation for the spatial experiment."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

from . import estimator as model
from . import primitives
from .reference import correlation_family, RANKS


def select_scope(split, responses, *, scan, rotation, orientation, outer_fold,
                 budget, arms):
    """Select on two context-only directions and every calibration subset.

    Each direction/subset contributes the mean over its finite pseudo-target
    correlations. Ties prefer larger decoder penalties, then pair penalties.
    """
    subsets = model.subset_ids_for(split, orientation, budget)
    ordinary = [arm for arm in arms if arm != 'D1_SHRUNK']
    accumulator = {arm: {key: [] for key in model.arm_grid(arm)} for arm in ordinary}
    reference_scores = {(rank, ridge): {} for rank in RANKS
                        for ridge in model.COVARIANCE_RIDGE_GRID}
    for direction in (0, 1):
        for subset in subsets:
            cell = model.build_pseudo_cell(
                split=split, responses=responses, scan=scan, rotation=rotation,
                orientation=orientation, outer_fold=outer_fold, budget=budget,
                subset_id=subset, direction=direction)
            relationship = model.cell_relationship_inputs(
                decoder_module=primitives, arrays=cell.arrays)
            if 'D1_SHRUNK' in arms:
                family = correlation_family(cell.arrays.calibration_target,
                    cell.arrays.calibration_context, cell.arrays.context_ids)
                order = np.argsort(np.asarray(cell.arrays.target_ids))
                for rank, correlations in family.items():
                    for ridge in model.COVARIANCE_RIDGE_GRID:
                        prediction = model.decode_arm(decoder_module=primitives,
                            arm='D1_SHRUNK', predicted=correlations,
                            relationship=relationship, arrays=cell.arrays,
                            covariance_ridge=ridge, context_block=cell.arrays.test_context)
                        r = model.columnwise_pearson(cell.validation_target, prediction)[order]
                        if not np.isfinite(r).all():
                            raise ValueError('Reference selection requires finite paired scores')
                        reference_scores[rank, ridge][subset, direction] = math.fsum(r.tolist()) / len(r)
            for arm in ordinary:
                cache = {}
                for pair_ridge, covariance_ridge in model.arm_grid(arm):
                    if pair_ridge not in cache:
                        cache[pair_ridge] = model.predicted_correlations_for_arm(
                            decoder_module=primitives, arm=arm,
                            relationship=relationship, arrays=cell.arrays,
                            pair_ridge=pair_ridge)
                    prediction = model.decode_arm(
                        decoder_module=primitives, arm=arm,
                        predicted=cache[pair_ridge], relationship=relationship,
                        arrays=cell.arrays, covariance_ridge=covariance_ridge,
                        context_block=cell.arrays.test_context)
                    r = model.columnwise_pearson(cell.validation_target, prediction)
                    finite = r[np.isfinite(r)]
                    if not len(finite):
                        raise ValueError(f"No finite validation targets for {arm}")
                    accumulator[arm][pair_ridge, covariance_ridge].append(float(finite.mean()))
    means = {arm: {key: float(np.mean(values)) for key, values in grid.items()}
             for arm, grid in accumulator.items()}
    selected = model.select_hyperparameters(means)
    if 'D1_SHRUNK' in arms:
        reference_means = {}
        for key, scores in reference_scores.items():
            subset_means = [math.fsum([scores[subset, 0], scores[subset, 1]]) / 2
                            for subset in subsets]
            reference_means[key] = math.fsum(subset_means) / len(subset_means)
        rank, ridge = min(reference_means, key=lambda key: (
            -reference_means[key], 10**9 if key[0] == 'full' else int(key[0]), -key[1]))
        selected['D1_SHRUNK'] = {'pair_ridge': None, 'covariance_ridge': ridge,
            'rank_label': rank, 'mean_pseudo_validation_pearson': reference_means[rank, ridge]}
        means['D1_SHRUNK'] = reference_means
    return selected, means


def predict_cell(split, responses, identity, settings):
    """Predict all requested arms before any target test labels are accessed."""
    arrays = model.build_cell_arrays(split=split, responses=responses, identity=identity)
    relationship = model.cell_relationship_inputs(decoder_module=primitives, arrays=arrays)
    predictions = {}
    for arm, penalties in settings.items():
        if arm == 'D1_SHRUNK':
            rank = penalties['rank_label']
            correlations = correlation_family(arrays.calibration_target,
                arrays.calibration_context, arrays.context_ids, ranks=(rank,))[rank]
        else:
            correlations = model.predicted_correlations_for_arm(
                decoder_module=primitives, arm=arm, relationship=relationship,
                arrays=arrays, pair_ridge=penalties['pair_ridge'])
        prediction = model.decode_arm(
            decoder_module=primitives, arm=arm, predicted=correlations,
            relationship=relationship, arrays=arrays,
            covariance_ridge=penalties['covariance_ridge'],
            context_block=arrays.test_context)
        # The study stored predictions as float32 before computing float64 scores.
        predictions[arm] = np.ascontiguousarray(prediction, dtype=np.float32)
    return predictions, arrays.target_ids


def score_cell(split, responses, identity, predictions, target_ids):
    """Score paired predictions on one common, label-defined target mask."""
    test_movie, _, _ = model.movie_roles_for_rotation(split, identity.movie_rotation)
    _, repeats = model.pool_repeats(split, identity.repeat_orientation)
    columns = np.array([int(x.split('::')[1]) for x in target_ids], dtype=np.int32)
    labels = model.stack_samples(responses, split.condition_axis[identity.scan],
                                 [test_movie], repeats, columns)
    centered = labels - labels.mean(axis=0, dtype=np.float64)[None, :]
    sst = np.sum(centered * centered, axis=0, dtype=np.float64)
    valid = np.all(np.isfinite(labels), axis=0) & (sst > 1e-12)
    if not valid.any():
        raise ValueError('No nonconstant finite test targets')
    rows = []
    for arm, stored in predictions.items():
        prediction = np.asarray(stored, dtype=np.float64)[:, valid]
        true = labels[:, valid]
        y = true - true.mean(axis=0, dtype=np.float64)[None, :]
        p = prediction - prediction.mean(axis=0, dtype=np.float64)[None, :]
        prediction_ss = np.sum(p * p, axis=0, dtype=np.float64)
        if np.any(prediction_ss <= 1e-12):
            raise ValueError(f'Constant prediction on a valid target: {arm}')
        covariance = np.sum(y * p, axis=0, dtype=np.float64)
        r = np.clip(covariance / np.sqrt(sst[valid] * prediction_ss), -1.0, 1.0)
        if not np.isfinite(r).all():
            raise ValueError(f'Nonfinite test score: {arm}')
        rows.append({**identity.as_dict(), 'test_movie': test_movie, 'candidate': arm,
                     'n_targets': int(valid.sum()), 'mean_target_pearson': float(r.mean())})
    return rows


def aggregate_cells(cells):
    """Equal-weight subsets -> orientations -> folds -> movies -> scans.

    A restricted run returns summaries for its requested scopes. It must not
    be interpreted as the complete paper aggregate.
    """
    frame = cells.copy()
    frame['session'] = frame.scan.str.extract(r'session_(\d+)_')[0].astype(int)
    key = ['candidate', 'policy_arm', 'budget']
    score = 'mean_target_pearson'
    step1 = frame.groupby(key + ['scan', 'session', 'test_movie', 'repeat_orientation', 'outer_fold'], as_index=False)[score].mean()
    step2 = step1.groupby(key + ['scan', 'session', 'test_movie', 'outer_fold'], as_index=False)[score].mean()
    step3 = step2.groupby(key + ['scan', 'session', 'test_movie'], as_index=False)[score].mean()
    scan = step3.groupby(key + ['scan', 'session'], as_index=False)[score].mean()
    session = scan.groupby(key + ['session'], as_index=False)[score].mean()
    return {'scan_curve': scan, 'session_curve': session,
            'equal_scan': scan.groupby(key, as_index=False)[score].mean(),
            'equal_session': session.groupby(key, as_index=False)[score].mean()}
