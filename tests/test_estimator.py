"""Scientific invariants: split isolation, pair identity, ridge and projection."""
import hashlib
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from spatial_priors import estimator as m, primitives as p
from spatial_priors.pipeline import select_scope, predict_cell, score_cell
from spatial_priors.reference import correlation_family


def synthetic_data():
    rng = np.random.default_rng(801)
    scan = 'session_1_scan_1'
    n = 18
    neurons = pd.DataFrame({'scan': scan, 'neuron_index': np.arange(n),
        'outer_target_fold': np.arange(n) % 3, 'field': np.arange(n) // 9,
        'brain_area': 'V1', 'unit_x_um': rng.uniform(0, 100, n),
        'unit_y_um': rng.uniform(0, 100, n), 'unit_z_um': rng.uniform(0, 100, n),
        'analysis_eligible': True})
    movies = pd.DataFrame({'rotation_id': 0,
        'condition_hash': ['m0', 'm1', 'm2', 'm3', 'm4', 'm5'],
        'movie_role': ['test', 'validation', 'train', 'train', 'train', 'train'],
        'train_rank': [-1, -1, 0, 1, 2, 3]})
    repeats = pd.DataFrame({'repeat_orientation': 0, 'repeat_id': range(10),
        'repeat_role': ['calibration' if x % 2 == 0 else 'population' for x in range(10)]})
    subsets = pd.DataFrame({'repeat_orientation': [0], 'budget_repeats_per_movie': [1],
        'subset_id': ['toy'], 'selected_repeat_ids': ['0']})
    context = neurons.loc[neurons.outer_target_fold != 0, 'neuron_index'].to_numpy()
    pair, _ = m.regenerate_outer_context_pairs(context, scan, 'm0', 0)
    rows = [{'scan': scan, 'rotation_id': 0, 'outer_neuron_fold': 0,
        'pair_set_role': 'outer_context_context', 'validation_direction': np.nan,
        'pair_indices_sha256': hashlib.sha256(pair.astype('<i4').tobytes()).hexdigest()}]
    for direction, pseudo_fold in enumerate((1, 2)):
        context_fold = 2 if pseudo_fold == 1 else 1
        indices = neurons.loc[neurons.outer_target_fold == context_fold, 'neuron_index'].to_numpy()
        pairs, _ = m.regenerate_pseudo_context_pairs(indices, scan, 'm0', 0, pseudo_fold)
        rows.append({'scan': scan, 'rotation_id': 0, 'outer_neuron_fold': 0,
            'pair_set_role': 'pseudo_context_context', 'validation_direction': direction,
            'pair_indices_sha256': hashlib.sha256(pairs.astype('<i4').tobytes()).hexdigest()})
    split = m.SplitInputs(Path('.'), neurons, movies, repeats, subsets,
        {scan: {f'm{i}': i for i in range(6)}}, pd.DataFrame(rows))
    common = rng.normal(size=(6, 10, 8, 1))
    responses = common + rng.normal(size=(6, 10, 8, n))
    return split, responses, scan


class EstimatorTests(unittest.TestCase):
    def test_ridge_intercept_is_not_penalized(self):
        x = np.arange(40, dtype=float).reshape(20, 2)
        fit = p.fit_pair_ridge(x, np.full(20, 7.0), alpha=1e9,
            feature_names=['a', 'b'], boolean_mask=[False, False])
        np.testing.assert_allclose(p.predict_pair_ridge(fit, x), 7.0, atol=1e-12)

    def test_covariance_decoder_matches_linear_system(self):
        covariance = np.array([[1.0, 0.4], [0.4, 1.0]])
        correlations = np.array([[0.2, 0.6]])
        # Check the system directly rather than a second implementation of solve.
        weights = p.covariance_decoder_weights(correlations, covariance, ridge=1.0)
        np.testing.assert_allclose(weights @ (covariance + np.eye(2)), correlations, atol=1e-15)
        gain = p.self_consistent_gain(correlations, weights, covariance)
        self.assertGreater(gain[0], 0)

    def test_configured_pair_sets_regenerate(self):
        split = m.load_split_inputs(ROOT / 'configs/splits')
        count = 0
        for scan in split.neuron_folds.scan.unique():
            neurons = m.scan_neurons(split, scan)
            for rotation in range(6):
                test, _, _ = m.movie_roles_for_rotation(split, rotation)
                for fold in range(3):
                    context = neurons.loc[neurons.outer_target_fold != fold, 'neuron_index'].to_numpy()
                    pairs, _ = m.regenerate_outer_context_pairs(context, scan, test, fold)
                    m.verify_pair_set(split, pairs, scan, rotation, fold)
                    count += 1
                    other = [f for f in range(3) if f != fold]
                    for direction in (0, 1):
                        indices = neurons.loc[neurons.outer_target_fold == other[1 - direction], 'neuron_index'].to_numpy()
                        pairs, _ = m.regenerate_pseudo_context_pairs(indices, scan, test, fold, other[direction])
                        m.verify_pseudo_pair_set(split, pairs, scan, rotation, fold, direction)
                        count += 1
        self.assertEqual(count, 540)

    def test_outer_targets_cannot_influence_selection(self):
        split, responses, scan = synthetic_data()
        kwargs = dict(scan=scan, rotation=0, orientation=0, outer_fold=0,
                      budget=1, arms=['D1', 'D1_PAIR_CALIBRATED', 'D1_DXY'])
        first = select_scope(split, responses, **kwargs)
        altered = responses.copy()
        altered[..., ::3] = 9999.0
        second = select_scope(split, altered, **kwargs)
        self.assertEqual(first, second)

    def test_test_targets_do_not_change_predictions(self):
        split, responses, scan = synthetic_data()
        identity = m.CellIdentity(scan, 0, 0, 0, 1, 'toy')
        settings = {'D1_DXY': {'pair_ridge': 5.0, 'covariance_ridge': 30.0}}
        first, ids = predict_cell(split, responses, identity, settings)
        altered = responses.copy()
        altered[0, :, :, ::3] = 9999.0
        second, _ = predict_cell(split, altered, identity, settings)
        np.testing.assert_array_equal(first['D1_DXY'], second['D1_DXY'])
        scored = score_cell(split, responses, identity, first, ids)
        self.assertEqual(scored[0]['n_targets'], 6)
        with self.assertRaises(ValueError):
            score_cell(split, altered, identity, second, ids)

    def test_guard_rejects_test_target_access(self):
        guard = m.TerminalGuard('test', frozenset({2}))
        with self.assertRaises(ValueError):
            guard.check(['test'], np.array([1, 2]))
        guard.check(['train'], np.array([2]))
        guard.check(['test'], np.array([1]))

    def test_reference_full_is_direct_and_rank_is_bounded(self):
        rng = np.random.default_rng(31)
        target = rng.normal(size=(60, 20))
        context = rng.normal(size=(60, 24))
        family = correlation_family(target, context, [str(i) for i in range(24)], ranks=('8', 'full'))
        expected = np.corrcoef(target.T, context.T)[:20, 20:]
        np.testing.assert_allclose(family['full'], expected, atol=1e-15)
        self.assertLessEqual(np.linalg.matrix_rank(family['8']), 8)
        with self.assertRaises(ValueError):
            correlation_family(target, context, [str(i) for i in range(24)])


if __name__ == '__main__':
    unittest.main()
