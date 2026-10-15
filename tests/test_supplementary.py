"""Scientific boundary and numerical checks for supplementary estimators."""
import hashlib
import json
import numpy as np
import unittest

from spatial_priors.supplementary import (
    ClassPairEncoder, OptionalFeatureScaler, fit_activity_profiles,
    fit_anchor_profiles, fit_locations, fit_stimulus_profiles,
    fixed_location_projection, location_pair_distances, normalize_profiles,
    profile_pair_features, select_anchor_positions, shared_input_features,
)



class SupplementaryTests(unittest.TestCase):
    def test_fixed_location_projection_matches_reference(self):
        matrix = fixed_location_projection()
        digest = hashlib.sha256()
        digest.update(matrix.dtype.str.encode('ascii'))
        digest.update(json.dumps(list(matrix.shape), separators=(',', ':')).encode('ascii'))
        digest.update(memoryview(np.ascontiguousarray(matrix)).cast('B'))
        assert digest.hexdigest() == '47f7fe6eae3f344fa932f31a87265f8404893bc393641f4307f4ed4dd5bc11b9'


    def test_location_recovers_known_cell_and_rejects_flat_response(self):
        rng = np.random.default_rng(42)
        features = np.zeros((40, 384, 32))
        features[:, 53] = rng.normal(size=(40, 32))
        response = np.column_stack([features[:, 53, 0], np.ones(40)])
        movies = np.repeat(['a', 'b', 'c', 'd'], 10)
        fit = fit_locations(features, response, movies)
        assert fit['selected_cell'][0] == 53
        assert fit['selected_row'][0] == 2 and fit['selected_column'][0] == 5
        assert fit['usable'].tolist() == [True, False]
        assert fit['leave_one_fold_out_rms'][0] == 0
        assert fit['selected_cell'][1] == 0  # Explicit invalid sentinel, never a valid RF.
        with self.assertRaisesRegex(ValueError, 'four training movies'):
            fit_locations(features, response, np.repeat(['a', 'b'], 20))


    def test_activity_basis_uses_context_only_and_constant_profiles_stay_zero(self):
        rng = np.random.default_rng(19)
        context = rng.normal(size=(70, 65))
        target = rng.normal(size=(70, 3))
        target[:, 2] = 3
        ids = [f'frame-{index:03}' for index in range(70)]
        first = fit_activity_profiles(target, context, ids)
        second = fit_activity_profiles(target * np.array([2, -1, 1]), context, ids)
        np.testing.assert_array_equal(first['basis'], second['basis'])
        np.testing.assert_array_equal(first['context_profiles'], second['context_profiles'])
        assert first['target_available'].tolist() == [True, True, False]
        np.testing.assert_array_equal(first['target_profiles'][2], 0)
        for column in range(64):
            pivot = np.argmax(np.abs(first['basis'][:, column]))
            assert first['basis'][pivot, column] > 0


    def test_anchor_selection_ignores_context_order_and_repeat_means_are_used(self):
        units = [f'unit-{index}' for index in range(70)]
        indices = list(range(70))
        selected = select_anchor_positions('session_5_scan_6', 0, indices, units)
        reverse = select_anchor_positions('session_5_scan_6', 0, indices[::-1], units[::-1])
        assert [units[i] for i in selected] == [units[::-1][i] for i in reverse]
        rng = np.random.default_rng(7)
        context = rng.normal(size=(4, 2, 10, 70))
        # Identical repeat means should give identical target profiles.
        target = context[:, :, :, :2].copy()
        changed = target.copy()
        changed[:, 0] += 20
        changed[:, 1] -= 20
        a = fit_anchor_profiles(target, context, selected)
        b = fit_anchor_profiles(changed, context, selected)
        np.testing.assert_allclose(a['target_correlations'], b['target_correlations'], atol=1e-14)


    def test_class_vocabulary_is_unordered_fitted_only_and_missing_is_explicit(self):
        labels = ['L2IT', 'L3IT', 'L4IT', None]
        encoder = ClassPairEncoder.fit(labels, np.array([[0, 1], [0, 3]]))
        assert encoder.vocabulary == (('L2IT', 'L3IT'),)
        encoded = encoder.transform(np.array([[1, 0], [0, 2], [2, 3]]))
        np.testing.assert_array_equal(encoded, [[1, 0, 0], [0, 0, 0], [0, 1, 0]])
        coarse = ClassPairEncoder.fit(labels, np.array([[0, 1]]), coarse=True)
        assert coarse.vocabulary == (('L23', 'L23'),)


    def test_shared_input_distinguishes_unknown_zero_norm_and_zero_overlap(self):
        counts = np.array([[3, 0], [0, 4], [0, 0], [3, 2]])
        known = [True, True, True, False]
        q, h = shared_input_features(counts, known, np.array([[0, 1], [0, 2], [0, 3]]))
        assert h[0, 0] == 0  # Defined absence of shared input.
        assert np.isnan(h[1:, 0]).all()  # Undefined and unknown are not zero.
        assert q[0, -1] == 1 and q[1, -1] == 0 and q[2, -1] == 0
        assert np.isfinite(q[1, :4]).all() and np.isnan(q[2, :4]).all()
        scaler = OptionalFeatureScaler.fit(np.column_stack([q, h]), binary=[False]*4+[True]*3+[False])
        transformed = scaler.transform(np.column_stack([q, h]))
        assert np.isfinite(transformed).all()
        np.testing.assert_array_equal(transformed[:, -1], 0)  # Only known overlap was constant zero.


    def test_readout_ties_choose_largest_alpha_without_using_anchor_labels(self):
        rng = np.random.default_rng(8)
        features = rng.normal(size=(40, 6))
        responses = np.full((40, 2), 2.)
        movies = np.repeat(['a', 'b', 'c', 'd'], 10)
        result = fit_stimulus_profiles(features, responses, movies, rng.normal(size=(64, 6)))
        np.testing.assert_array_equal(result['selected_alpha'], 1000.)
        np.testing.assert_array_equal(result['profiles'], 0)
        assert not result['available'].any()


    def test_profile_features_and_location_distance_match_definitions(self):
        profiles, _ = normalize_profiles([[1, 0, -1], [2, 0, -2]])
        features, binary = profile_pair_features([1, 1], ['V1', 'V1'], [[0, 0, 0], [3, 4, 2]],
                                                 np.array([[0, 1]]), profiles, profiles)
        np.testing.assert_allclose(features[0], [1, 1, 2, np.sqrt(29), 1, 1, 1, 1, 1], atol=1e-7)
        assert binary.tolist() == [True, True] + [False]*7
        np.testing.assert_array_equal(location_pair_distances([[1, 1], [4, 5]], np.array([[0, 1]])), [5])


if __name__ == "__main__":
    unittest.main()
