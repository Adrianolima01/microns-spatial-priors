"""Numerical operations for shared correlation maps and covariance decoding."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable, Sequence
import numpy as np

NUMERICAL_EPSILON = 1e-12
FEATURE_SCALE_FLOOR = 1e-8
PAIR_RIDGE_GRID = (1.0, 5.0, 25.0)
COVARIANCE_RIDGE_GRID = (10.0, 30.0, 100.0)

def _finite_f64(array: Any, *, name: str, ndim: int | None = None) -> np.ndarray:
    value = np.asarray(array, dtype=np.float64)
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, found {value.ndim}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains non-finite values")
    return value


def _pairs(array: Any, n_neurons: int | None = None) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim != 2 or value.shape[1] != 2:
        raise ValueError("pairs must have shape [n_pairs,2]")
    if value.dtype.kind not in "iu":
        raise ValueError("pairs must be integer-valued")
    value = value.astype(np.int64, copy=False)
    if np.any(value < 0):
        raise ValueError("pairs cannot contain negative indices")
    if n_neurons is not None and np.any(value >= int(n_neurons)):
        raise ValueError("pair index exceeds neuron count")
    if np.any(value[:, 0] == value[:, 1]):
        raise ValueError("self-pairs are forbidden")
    return value


def m0_pair_features(
    *,
    field: Sequence[Any],
    brain_area: Sequence[Any],
    xyz_um: Any,
    pairs: Any,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    """Construct the exact four-field no-response M0 pair map."""

    field_value = np.asarray(field, dtype=object)
    area_value = np.asarray(brain_area, dtype=object)
    xyz = _finite_f64(xyz_um, name="xyz_um", ndim=2)
    if xyz.shape[1] != 3:
        raise ValueError("xyz_um must have shape [n_neurons,3]")
    n_neurons = len(xyz)
    if len(field_value) != n_neurons or len(area_value) != n_neurons:
        raise ValueError("metadata row counts do not match xyz_um")
    pair_index = _pairs(pairs, n_neurons)
    left = pair_index[:, 0]
    right = pair_index[:, 1]
    same_field = (field_value[left] == field_value[right]).astype(np.float64)
    same_area = (area_value[left] == area_value[right]).astype(np.float64)
    difference = xyz[left] - xyz[right]
    depth_difference = np.abs(difference[:, 2])
    xyz_distance = np.sqrt(np.sum(difference * difference, axis=1))
    features = np.column_stack(
        [same_field, same_area, depth_difference, xyz_distance]
    )
    names = (
        "same_field",
        "same_brain_area",
        "absolute_soma_depth_difference_um",
        "soma_xyz_distance_um",
    )
    boolean_mask = np.asarray([True, True, False, False], dtype=bool)
    return features.astype(np.float64), names, boolean_mask


@dataclass(frozen=True)
class PairRidgeModel:
    alpha: float
    feature_names: tuple[str, ...]
    boolean_mask: np.ndarray
    continuous_mean: np.ndarray
    continuous_scale: np.ndarray
    intercept: float
    coefficients: np.ndarray
    solver: str


def _normalize_pair_features(
    features: Any,
    boolean_mask: Any,
    *,
    fit: bool,
    mean: Any | None = None,
    scale: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = _finite_f64(features, name="features", ndim=2)
    boolean = np.asarray(boolean_mask, dtype=bool)
    if boolean.shape != (x.shape[1],):
        raise ValueError("boolean_mask width does not match features")
    if np.any((x[:, boolean] != 0.0) & (x[:, boolean] != 1.0)):
        raise ValueError("boolean feature columns must contain exact 0/1")
    continuous = ~boolean
    if fit:
        fitted_mean = np.zeros(x.shape[1], dtype=np.float64)
        fitted_scale = np.ones(x.shape[1], dtype=np.float64)
        if np.any(continuous):
            fitted_mean[continuous] = x[:, continuous].mean(axis=0)
            fitted_scale[continuous] = x[:, continuous].std(axis=0, ddof=0)
            bad = (
                (~np.isfinite(fitted_scale))
                | (fitted_scale < FEATURE_SCALE_FLOOR)
            )
            fitted_scale[bad] = 1.0
    else:
        fitted_mean = _finite_f64(mean, name="continuous_mean", ndim=1)
        fitted_scale = _finite_f64(scale, name="continuous_scale", ndim=1)
        if fitted_mean.shape != (x.shape[1],) or fitted_scale.shape != (
            x.shape[1],
        ):
            raise ValueError("saved normalization width does not match features")
        if np.any(fitted_scale <= 0.0):
            raise ValueError("saved feature scales must be positive")
    normalized = x.copy()
    normalized[:, continuous] = (
        x[:, continuous] - fitted_mean[continuous]
    ) / fitted_scale[continuous]
    return normalized, fitted_mean, fitted_scale


def fit_pair_ridge(
    features: Any,
    targets: Any,
    *,
    alpha: float,
    feature_names: Iterable[str],
    boolean_mask: Any,
) -> PairRidgeModel:
    """Fit ridge with an exactly unpenalized intercept."""

    if float(alpha) < 0.0 or not np.isfinite(float(alpha)):
        raise ValueError("alpha must be finite and nonnegative")
    x, mean, scale = _normalize_pair_features(
        features, boolean_mask, fit=True
    )
    y = _finite_f64(targets, name="targets", ndim=1)
    if len(y) != len(x):
        raise ValueError("target length does not match feature rows")
    names = tuple(str(value) for value in feature_names)
    if len(names) != x.shape[1] or len(set(names)) != len(names):
        raise ValueError("feature_names must be unique and match feature width")
    design = np.column_stack([np.ones(len(x), dtype=np.float64), x])
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    gram = design.T @ design + penalty
    rhs = design.T @ y
    try:
        beta = np.linalg.solve(gram, rhs)
        solver = "solve"
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(gram) @ rhs
        solver = "pinv"
    if not np.all(np.isfinite(beta)):
        raise RuntimeError("pair ridge produced non-finite coefficients")
    return PairRidgeModel(
        alpha=float(alpha),
        feature_names=names,
        boolean_mask=np.asarray(boolean_mask, dtype=bool).copy(),
        continuous_mean=mean,
        continuous_scale=scale,
        intercept=float(beta[0]),
        coefficients=beta[1:].astype(np.float64),
        solver=solver,
    )


def predict_pair_ridge(model: PairRidgeModel, features: Any) -> np.ndarray:
    x, _, _ = _normalize_pair_features(
        features,
        model.boolean_mask,
        fit=False,
        mean=model.continuous_mean,
        scale=model.continuous_scale,
    )
    if x.shape[1] != len(model.coefficients):
        raise ValueError("feature width does not match fitted model")
    prediction = model.intercept + x @ model.coefficients
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError("pair ridge prediction contains non-finite values")
    return prediction.astype(np.float64)


@dataclass(frozen=True)
class PopulationStandardization:
    mean: np.ndarray
    scale: np.ndarray


def fit_population_standardization(
    observations: Any,
) -> tuple[np.ndarray, PopulationStandardization]:
    value = _finite_f64(observations, name="observations", ndim=2)
    if len(value) < 2:
        raise ValueError("at least two population observations are required")
    mean = value.mean(axis=0)
    scale = value.std(axis=0, ddof=0)
    bad = (~np.isfinite(scale)) | (scale < FEATURE_SCALE_FLOOR)
    scale[bad] = 1.0
    standardized = (value - mean[None, :]) / scale[None, :]
    return standardized, PopulationStandardization(mean=mean, scale=scale)


def apply_population_standardization(
    observations: Any, normalization: PopulationStandardization
) -> np.ndarray:
    value = _finite_f64(observations, name="observations", ndim=2)
    if value.shape[1] != len(normalization.mean):
        raise ValueError("population width does not match normalization")
    return (
        (value - normalization.mean[None, :])
        / normalization.scale[None, :]
    ).astype(np.float64)


def pair_correlations_standardized(
    standardized_observations: Any,
    pairs: Any,
    *,
    chunk_size: int = 20_000,
) -> np.ndarray:
    """Compute population Pearson correlations from already standardized rows."""

    z = _finite_f64(
        standardized_observations,
        name="standardized_observations",
        ndim=2,
    )
    pair_index = _pairs(pairs, z.shape[1])
    if int(chunk_size) < 1:
        raise ValueError("chunk_size must be positive")
    output = np.empty(len(pair_index), dtype=np.float64)
    denominator = float(len(z))
    for start in range(0, len(pair_index), int(chunk_size)):
        stop = min(start + int(chunk_size), len(pair_index))
        left = pair_index[start:stop, 0]
        right = pair_index[start:stop, 1]
        output[start:stop] = np.sum(z[:, left] * z[:, right], axis=0)
        output[start:stop] /= denominator
    return np.clip(output, -1.0, 1.0)


def target_context_correlations(
    target_observations: Any,
    context_observations: Any,
) -> np.ndarray:
    """Compute D1/D2 target-to-context calibration correlations."""

    target = _finite_f64(
        target_observations, name="target_observations", ndim=2
    )
    context = _finite_f64(
        context_observations, name="context_observations", ndim=2
    )
    if len(target) != len(context) or len(target) < 2:
        raise ValueError("target/context calibration rows must match and exceed one")
    target_z, _ = fit_population_standardization(target)
    context_z, _ = fit_population_standardization(context)
    correlation = target_z.T @ context_z / float(len(target_z))
    return np.clip(correlation, -1.0, 1.0).astype(np.float64)


def context_correlation_matrix(
    standardized_context: Any,
) -> np.ndarray:
    z = _finite_f64(
        standardized_context, name="standardized_context", ndim=2
    )
    covariance = z.T @ z / float(len(z))
    covariance = 0.5 * (covariance + covariance.T)
    covariance[np.diag_indices_from(covariance)] = 1.0
    return covariance.astype(np.float64)


def covariance_decoder_weights(
    predicted_correlations: Any,
    context_correlation: Any,
    *,
    ridge: float,
) -> np.ndarray:
    correlations = _finite_f64(
        predicted_correlations, name="predicted_correlations"
    )
    if correlations.ndim == 1:
        correlations = correlations[None, :]
    if correlations.ndim != 2:
        raise ValueError("predicted_correlations must be [target,context]")
    covariance = _finite_f64(
        context_correlation, name="context_correlation", ndim=2
    )
    if covariance.shape[0] != covariance.shape[1]:
        raise ValueError("context_correlation must be square")
    if correlations.shape[1] != covariance.shape[0]:
        raise ValueError("correlation/context widths do not match")
    if not np.allclose(covariance, covariance.T, atol=1e-10, rtol=1e-10):
        raise ValueError("context_correlation must be symmetric")
    if float(ridge) < 0.0 or not np.isfinite(float(ridge)):
        raise ValueError("ridge must be finite and nonnegative")
    regularized = covariance + float(ridge) * np.eye(
        len(covariance), dtype=np.float64
    )
    try:
        weights = np.linalg.solve(regularized, correlations.T).T
    except np.linalg.LinAlgError:
        weights = correlations @ np.linalg.pinv(regularized)
    if not np.all(np.isfinite(weights)):
        raise RuntimeError("covariance decoder produced non-finite weights")
    return weights.astype(np.float64)


def self_consistent_gain(
    predicted_correlations: Any,
    weights: Any,
    context_correlation: Any,
    *,
    epsilon: float = NUMERICAL_EPSILON,
) -> np.ndarray:
    correlations = _finite_f64(
        predicted_correlations, name="predicted_correlations"
    )
    decoder_weights = _finite_f64(weights, name="weights")
    if correlations.ndim == 1:
        correlations = correlations[None, :]
    if decoder_weights.ndim == 1:
        decoder_weights = decoder_weights[None, :]
    if correlations.shape != decoder_weights.shape:
        raise ValueError("predicted correlations and weights must have same shape")
    covariance = _finite_f64(
        context_correlation, name="context_correlation", ndim=2
    )
    if covariance.shape != (
        correlations.shape[1],
        correlations.shape[1],
    ):
        raise ValueError("context covariance shape does not match weights")
    numerator = np.sum(correlations * decoder_weights, axis=1)
    denominator = np.einsum(
        "ti,ij,tj->t", decoder_weights, covariance, decoder_weights
    )
    gain = numerator / (denominator + float(epsilon))
    gain[(~np.isfinite(gain)) | (denominator <= float(epsilon))] = 0.0
    return gain.astype(np.float64)


def decode_test_context(
    test_context_observations: Any,
    context_normalization: PopulationStandardization,
    predicted_correlations: Any,
    context_correlation: Any,
    *,
    covariance_ridge: float,
    target_calibration_mean: Any | None = None,
    target_calibration_scale: Any | None = None,
    epsilon: float = NUMERICAL_EPSILON,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Decode held-out targets without reading their labels.

    If calibration-only target mean/scale are supplied, predictions are
    returned on raw target-response scale for R2.  Otherwise standardized
    predictions are returned, which is sufficient for Pearson scoring.
    """

    test_context_z = apply_population_standardization(
        test_context_observations, context_normalization
    )
    correlations = _finite_f64(
        predicted_correlations, name="predicted_correlations"
    )
    if correlations.ndim == 1:
        correlations = correlations[None, :]
    weights = covariance_decoder_weights(
        correlations, context_correlation, ridge=covariance_ridge
    )
    gain = self_consistent_gain(
        correlations,
        weights,
        context_correlation,
        epsilon=epsilon,
    )
    prediction_z = (test_context_z @ weights.T) * gain[None, :]
    if (target_calibration_mean is None) != (
        target_calibration_scale is None
    ):
        raise ValueError("target calibration mean and scale must be supplied together")
    if target_calibration_mean is None:
        prediction = prediction_z
    else:
        target_mean = _finite_f64(
            target_calibration_mean,
            name="target_calibration_mean",
            ndim=1,
        )
        target_scale = _finite_f64(
            target_calibration_scale,
            name="target_calibration_scale",
            ndim=1,
        )
        if len(target_mean) != prediction_z.shape[1] or len(
            target_scale
        ) != prediction_z.shape[1]:
            raise ValueError("target scale vectors do not match target count")
        safe_scale = target_scale.copy()
        safe_scale[safe_scale < FEATURE_SCALE_FLOOR] = 1.0
        prediction = target_mean[None, :] + prediction_z * safe_scale[None, :]
    return prediction.astype(np.float64), {
        "weights": weights,
        "gain": gain,
        "prediction_standardized": prediction_z.astype(np.float64),
    }
