"""Matched direct, shared-map, spatial and within-field prediction estimators.

Splits and pair identities are explicit inputs. Fitting uses training responses
and context-only pseudo-target validation. Test targets enter only scoring.
"""
from __future__ import annotations
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd
from . import primitives

BUDGETS = (1, 2, 4, 5)
OUTER_FOLDS = (0, 1, 2)
PAIR_RIDGE_GRID = (1.0, 5.0, 25.0)
COVARIANCE_RIDGE_GRID = (10.0, 30.0, 100.0)
MAX_TRAIN_PAIRS = 6000
PHASE8_SEED = 20260725
NUMERICAL_EPSILON = 1e-12
MASTER_ORDER = ('same_field', 'same_brain_area', 'DZ', 'D3D', 'DXY', 'RAW_D1')
MASTER_BOOLEAN = (True, True, False, False, False, False)
ARM_INDICES = {
    'D1': None,
    'D1_PAIR_CALIBRATED': (5,),
    'D1_DZ': (2, 5),
    'D1_DXY': (4, 5),
    'D1_D3D': (3, 5),
    'D1_DZ_DXY': (2, 4, 5),
    'D1_DZ_D3D_LEGACY': (2, 3, 5),
    'WF_D1': None,
    'WF_D1_PAIR_CALIBRATED': (5,),
    'WF_D1_DXY': (4, 5),
    'WF_D1_DZ_D3D_LEGACY': (2, 3, 5),
}
ARMS = tuple(ARM_INDICES)
RAW_ARMS = ('D1', 'WF_D1')

class SpatialDecompositionError(ValueError):
    """Invalid data or a violated split/model assumption."""


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()

def stable_seed(*parts: Any) -> int:
    key = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 2147483647


@dataclass(frozen=True)
class SplitInputs:
    root: Path
    neuron_folds: pd.DataFrame
    movie_roles: pd.DataFrame
    repeat_pools: pd.DataFrame
    budget_subsets: pd.DataFrame
    condition_axis: Mapping[str, Mapping[str, int]]
    pair_set_manifest: pd.DataFrame


def load_split_inputs(split_root: Path) -> SplitInputs:
    nf = pd.read_csv(split_root / "neuron_folds.csv", float_precision="round_trip")
    nf = nf[nf["analysis_eligible"].astype(bool)].copy()
    mr = pd.read_csv(split_root / "movie_roles.csv")
    rp = pd.read_csv(split_root / "repeat_pools.csv")
    bs = pd.read_csv(split_root / "budget_subsets.csv")
    sr = pd.read_csv(
        split_root / "condition_axis.csv",
        usecols=["scan", "condition_hash", "local_condition_axis"],
    ).drop_duplicates()
    axis: dict[str, dict[str, int]] = {}
    for scan, group in sr.groupby("scan"):
        axis[str(scan)] = {
            str(r.condition_hash): int(r.local_condition_axis)
            for r in group.itertuples()
        }
    pm = pd.read_csv(split_root / "pair_set_manifest.csv")
    return SplitInputs(split_root, nf, mr, rp, bs, MappingProxyType(axis), pm)


def movie_roles_for_rotation(split: SplitInputs, rotation: int) -> tuple[str, str, tuple[str, ...]]:
    rows = split.movie_roles[split.movie_roles["rotation_id"] == rotation]
    test = str(rows[rows["movie_role"] == "test"]["condition_hash"].iloc[0])
    validation = str(rows[rows["movie_role"] == "validation"]["condition_hash"].iloc[0])
    train = tuple(
        str(h) for h in
        rows[rows["movie_role"] == "train"].sort_values("train_rank")["condition_hash"]
    )
    if len(train) != 4:
        raise SpatialDecompositionError("rotation must have exactly four training movies")
    return test, validation, train


def pool_repeats(split: SplitInputs, orientation: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    rows = split.repeat_pools[split.repeat_pools["repeat_orientation"] == orientation]
    cal = tuple(int(v) for v in rows[rows["repeat_role"] == "calibration"]["repeat_id"])
    pop = tuple(int(v) for v in rows[rows["repeat_role"] == "population"]["repeat_id"])
    if len(cal) != 5 or len(pop) != 5 or set(cal) & set(pop):
        raise SpatialDecompositionError("repeat pools must be disjoint fives")
    return cal, pop


def subset_ids_for(split: SplitInputs, orientation: int, budget: int) -> tuple[str, ...]:
    rows = split.budget_subsets[
        (split.budget_subsets["repeat_orientation"] == orientation)
        & (split.budget_subsets["budget_repeats_per_movie"] == budget)
    ]
    return tuple(str(v) for v in rows["subset_id"])


def subset_repeats(split: SplitInputs, orientation: int, subset_id: str) -> tuple[int, ...]:
    rows = split.budget_subsets[
        (split.budget_subsets["repeat_orientation"] == orientation)
        & (split.budget_subsets["subset_id"] == subset_id)
    ]
    if len(rows) != 1:
        raise SpatialDecompositionError(f"subset id is not unique: {subset_id}")
    raw = str(rows["selected_repeat_ids"].iloc[0])
    return tuple(int(v) for v in raw.split(","))


def scan_neurons(split: SplitInputs, scan: str) -> pd.DataFrame:
    rows = split.neuron_folds[split.neuron_folds["scan"] == scan]
    return rows.sort_values("neuron_index").reset_index(drop=True)


def load_scan_responses(replication_root: Path, scan: str) -> np.ndarray:
    path = replication_root / scan / "processed" / "Y_clip_fluorescence_raw.npy"
    if not path.is_file():
        raise SpatialDecompositionError(f"missing response array: {path}")
    # Memory-mapped so only the slices a lawful role requests are ever paged in.
    # np.load without mmap_mode materialises the sealed terminal-label region too.
    array = np.load(path, mmap_mode="r")
    if array.ndim != 4:
        raise SpatialDecompositionError("response array must be [movie,repeat,frame,neuron]")
    return array


@dataclass(frozen=True)
class TerminalGuard:
    """Refuses any selection that crosses the test movie with an outer target neuron.

    Only scoring may enable access to held-out target labels. Fitting and
    model selection retain the default restriction.
    """

    test_condition_hash: str
    target_indices: frozenset[int]
    licensed: bool = False

    def check(self, condition_hashes: Sequence[str], neuron_columns: np.ndarray) -> None:
        if self.licensed:
            return
        if self.test_condition_hash not in set(condition_hashes):
            return
        overlap = self.target_indices.intersection(int(v) for v in neuron_columns)
        if overlap:
            raise SpatialDecompositionError(
                "refusing to read held-out target test labels: selection crosses "
                f"test movie {self.test_condition_hash} with {len(overlap)} outer target "
                "neurons and no explicit scoring access is held"
            )


def stack_samples(
    responses: np.ndarray,
    axis_of: Mapping[str, int],
    condition_hashes: Sequence[str],
    repeats: Sequence[int],
    neuron_columns: np.ndarray,
    *,
    guard: "TerminalGuard | None" = None,
) -> np.ndarray:
    if guard is not None:
        guard.check(condition_hashes, neuron_columns)
    blocks = [
        responses[axis_of[h]][r][:, neuron_columns]
        for h in condition_hashes
        for r in repeats
    ]
    stacked = np.concatenate(blocks, axis=0).astype(np.float64)
    if not np.isfinite(stacked).all():
        raise SpatialDecompositionError("response block contains non-finite values")
    return stacked


def dxy_pair_feature(xyz_um: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """In-plane soma-coordinate distance."""
    xyz = np.asarray(xyz_um, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise SpatialDecompositionError("xyz_um must be [n_neurons,3]")
    index = np.asarray(pairs, dtype=np.int64)
    if index.ndim != 2 or index.shape[1] != 2:
        raise SpatialDecompositionError("pairs must be [n_pairs,2]")
    difference = xyz[index[:, 0]] - xyz[index[:, 1]]
    value = np.sqrt(difference[:, 0] ** 2 + difference[:, 1] ** 2)
    if not np.isfinite(value).all():
        raise SpatialDecompositionError("DXY contains non-finite values")
    return value.astype(np.float64)


def build_master_map(
    *,
    decoder_module: ModuleType,
    field: Sequence[Any],
    brain_area: Sequence[Any],
    xyz_um: np.ndarray,
    pairs: np.ndarray,
    raw_d1_column: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Six columns: M0 (0-3), local DXY (4), RAW_D1 (5).

    Columns 0 to 3 are field/area indicators and depth/3D distances,
    supplied by ``m0_pair_features``.
    """
    m0, names, boolean = decoder_module.m0_pair_features(
        field=field, brain_area=brain_area, xyz_um=xyz_um, pairs=pairs
    )
    expected = (
        "same_field", "same_brain_area",
        "absolute_soma_depth_difference_um", "soma_xyz_distance_um",
    )
    if tuple(names) != expected:
        raise SpatialDecompositionError(
            "installed geometry feature names drifted; refusing to proceed"
        )
    if tuple(np.asarray(boolean, dtype=bool).tolist()) != (True, True, False, False):
        raise SpatialDecompositionError("geometry boolean mask drifted")
    dxy = dxy_pair_feature(xyz_um, pairs)
    raw = np.asarray(raw_d1_column, dtype=np.float64).reshape(-1)
    if len(raw) != len(m0):
        raise SpatialDecompositionError("RAW_D1 column length does not match pair rows")
    master = np.column_stack([m0, dxy, raw]).astype(np.float64)
    if master.shape[1] != 6:
        raise SpatialDecompositionError("master map must have exactly six columns")
    return master, np.asarray(MASTER_BOOLEAN, dtype=bool)


def slice_arm(master: np.ndarray, master_boolean: np.ndarray, arm: str) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    indices = ARM_INDICES[arm]
    if indices is None:
        raise SpatialDecompositionError(f"{arm} is a raw arm and has no feature slice")
    columns = np.asarray(indices, dtype=np.int64)
    return (
        np.ascontiguousarray(master[:, columns], dtype=np.float64),
        np.asarray(master_boolean, dtype=bool)[columns],
        tuple(MASTER_ORDER[i] for i in indices),
    )


def regenerate_outer_context_pairs(
    context_indices: np.ndarray, scan: str, test_hash: str, outer_fold: int
) -> tuple[np.ndarray, int]:
    """Deterministically sample context pairs for an outer target fold."""
    indices = np.asarray(np.sort(context_indices), dtype=np.int32)
    seed = stable_seed(PHASE8_SEED, scan, test_hash, outer_fold, "phase8d_pairs")
    left, right = np.triu_indices(len(indices), k=1)
    pairs = np.column_stack([indices[left], indices[right]]).astype(np.int32)
    if len(pairs) > MAX_TRAIN_PAIRS:
        rng = np.random.default_rng(seed)
        chosen = np.sort(rng.choice(len(pairs), size=MAX_TRAIN_PAIRS, replace=False))
        pairs = pairs[chosen]
    return pairs, seed


def verify_pair_set(
    split: SplitInputs, pairs: np.ndarray, scan: str, rotation: int, outer_fold: int
) -> str:
    digest = sha256_bytes(pairs.astype("<i4", copy=False).tobytes(order="C"))
    rows = split.pair_set_manifest[
        (split.pair_set_manifest["scan"] == scan)
        & (split.pair_set_manifest["rotation_id"] == rotation)
        & (split.pair_set_manifest["outer_neuron_fold"] == outer_fold)
        & (split.pair_set_manifest["pair_set_role"] == "outer_context_context")
    ]
    if len(rows) != 1:
        raise SpatialDecompositionError("outer context-context pair set is not unique")
    expected = str(rows["pair_indices_sha256"].iloc[0])
    if digest != expected:
        raise SpatialDecompositionError(
            f"pair set hash mismatch for {scan} rotation {rotation} fold {outer_fold}: "
            f"regenerated {digest} expected {expected}"
        )
    return digest


def target_context_pairs(target_indices: np.ndarray, context_indices: np.ndarray) -> np.ndarray:
    targets = np.repeat(np.asarray(target_indices, dtype=np.int32), len(context_indices))
    contexts = np.tile(np.asarray(context_indices, dtype=np.int32), len(target_indices))
    low = np.minimum(targets, contexts)
    high = np.maximum(targets, contexts)
    return np.column_stack([low, high]).astype(np.int32)


@dataclass(frozen=True)
class CellIdentity:
    scan: str
    movie_rotation: int
    repeat_orientation: int
    outer_fold: int
    budget: int
    calibration_subset_id: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "scan": self.scan,
            "movie_rotation": self.movie_rotation,
            "repeat_orientation": self.repeat_orientation,
            "outer_fold": self.outer_fold,
            "budget": self.budget,
            "calibration_subset_id": self.calibration_subset_id,
        }


@dataclass(frozen=True)
class CellArrays:
    target_ids: tuple[str, ...]
    context_ids: tuple[str, ...]
    calibration_target: np.ndarray
    calibration_context: np.ndarray
    pair_label_context: np.ndarray
    test_context: np.ndarray
    train_pairs_global: np.ndarray
    train_pairs_local: np.ndarray
    target_context_pairs_global: np.ndarray
    same_field_train_mask: np.ndarray
    per_target_field_context: Mapping[int, np.ndarray]
    field_of_target: np.ndarray
    xyz_um: np.ndarray
    field: np.ndarray
    brain_area: np.ndarray
    neuron_index_to_row: Mapping[int, int]


def build_cell_arrays(
    *, split: SplitInputs, responses: np.ndarray, identity: CellIdentity
) -> CellArrays:
    scan = identity.scan
    axis_of = split.condition_axis[scan]
    test_hash, _validation_hash, train_hashes = movie_roles_for_rotation(
        split, identity.movie_rotation
    )
    _cal_pool, pop_pool = pool_repeats(split, identity.repeat_orientation)
    cal_repeats = subset_repeats(split, identity.repeat_orientation, identity.calibration_subset_id)
    if len(cal_repeats) != identity.budget:
        raise SpatialDecompositionError("calibration subset size does not match the budget")

    neurons = scan_neurons(split, scan)
    target_rows = neurons[neurons["outer_target_fold"] == identity.outer_fold]
    context_rows = neurons[neurons["outer_target_fold"] != identity.outer_fold]
    target_index = np.sort(target_rows["neuron_index"].to_numpy(np.int32))
    context_index = np.sort(context_rows["neuron_index"].to_numpy(np.int32))

    xyz = neurons[["unit_x_um", "unit_y_um", "unit_z_um"]].to_numpy(np.float64)
    field = neurons["field"].to_numpy()
    area = neurons["brain_area"].to_numpy()
    row_of = {int(n): i for i, n in enumerate(neurons["neuron_index"].to_numpy())}

    # Unlicensed: this builder may never touch a sealed terminal evaluation label.
    guard = TerminalGuard(test_hash, frozenset(int(v) for v in target_index), licensed=False)
    cal_target = stack_samples(responses, axis_of, train_hashes, cal_repeats, target_index,
                               guard=guard)
    cal_context = stack_samples(responses, axis_of, train_hashes, cal_repeats, context_index,
                                guard=guard)
    pair_context = stack_samples(responses, axis_of, train_hashes, pop_pool, context_index,
                                 guard=guard)
    test_context = stack_samples(responses, axis_of, [test_hash], pop_pool, context_index,
                                 guard=guard)

    train_pairs, _seed = regenerate_outer_context_pairs(
        context_index, scan, test_hash, identity.outer_fold
    )
    verify_pair_set(split, train_pairs, scan, identity.movie_rotation, identity.outer_fold)
    local_of = {int(n): i for i, n in enumerate(context_index)}
    train_pairs_local = np.column_stack([
        np.asarray([local_of[int(v)] for v in train_pairs[:, 0]], dtype=np.int64),
        np.asarray([local_of[int(v)] for v in train_pairs[:, 1]], dtype=np.int64),
    ])
    tc_pairs = target_context_pairs(target_index, context_index)

    field_by_neuron = {int(n): field[row_of[int(n)]] for n in neurons["neuron_index"].to_numpy()}
    same_field_train = np.asarray(
        [field_by_neuron[int(a)] == field_by_neuron[int(b)] for a, b in train_pairs],
        dtype=bool,
    )
    if not same_field_train.any():
        raise SpatialDecompositionError("within-field pair training set is empty")

    per_target_field_context: dict[int, np.ndarray] = {}
    for position, neuron in enumerate(target_index):
        same = np.flatnonzero(
            np.asarray([field_by_neuron[int(c)] == field_by_neuron[int(neuron)]
                        for c in context_index], dtype=bool)
        )
        per_target_field_context[position] = same.astype(np.int64)
    retained = [p for p, v in per_target_field_context.items() if len(v) >= 1]
    if len(retained) != len(target_index):
        raise SpatialDecompositionError(
            "a target has no lawful same-field context neuron; the specified within-field "
            "construction retains all targets, so this indicates an input drift"
        )

    ids = lambda arr: tuple(f"{scan}::{int(v)}" for v in arr)
    return CellArrays(
        target_ids=ids(target_index),
        context_ids=ids(context_index),
        calibration_target=cal_target,
        calibration_context=cal_context,
        pair_label_context=pair_context,
        test_context=test_context,
        train_pairs_global=train_pairs,
        train_pairs_local=train_pairs_local,
        target_context_pairs_global=tc_pairs,
        same_field_train_mask=same_field_train,
        per_target_field_context=MappingProxyType(per_target_field_context),
        field_of_target=np.asarray([field_by_neuron[int(n)] for n in target_index]),
        xyz_um=xyz,
        field=field,
        brain_area=area,
        neuron_index_to_row=MappingProxyType(row_of),
    )


def cell_relationship_inputs(
    *, decoder_module: ModuleType, arrays: CellArrays
) -> dict[str, Any]:
    """Standardization, Sigma_C, Pool-P pair labels, raw D1, and the master maps."""
    pair_z, context_normalization = decoder_module.fit_population_standardization(
        arrays.pair_label_context
    )
    pair_labels = decoder_module.pair_correlations_standardized(
        pair_z, arrays.train_pairs_local
    )
    context_correlation = decoder_module.context_correlation_matrix(pair_z)
    raw_d1 = decoder_module.target_context_correlations(
        arrays.calibration_target, arrays.calibration_context
    )

    cal_z, _ = decoder_module.fit_population_standardization(arrays.calibration_context)
    raw_context_column = decoder_module.pair_correlations_standardized(
        cal_z, arrays.train_pairs_local
    )
    raw_target_column = np.ascontiguousarray(raw_d1.reshape(-1, order="C"), dtype=np.float64)

    # global neuron index -> row in the per-scan neuron table
    row_of = arrays.neuron_index_to_row
    remap = lambda pairs: np.column_stack([
        np.asarray([row_of[int(v)] for v in pairs[:, 0]], dtype=np.int64),
        np.asarray([row_of[int(v)] for v in pairs[:, 1]], dtype=np.int64),
    ])

    master_train, master_boolean = build_master_map(
        decoder_module=decoder_module,
        field=arrays.field, brain_area=arrays.brain_area, xyz_um=arrays.xyz_um,
        pairs=remap(arrays.train_pairs_global), raw_d1_column=raw_context_column,
    )
    master_target, _ = build_master_map(
        decoder_module=decoder_module,
        field=arrays.field, brain_area=arrays.brain_area, xyz_um=arrays.xyz_um,
        pairs=remap(arrays.target_context_pairs_global), raw_d1_column=raw_target_column,
    )
    return {
        "context_normalization": context_normalization,
        "pair_labels": pair_labels,
        "context_correlation": context_correlation,
        "raw_d1": raw_d1,
        "master_train": master_train,
        "master_target": master_target,
        "master_boolean": master_boolean,
        "n_targets": len(arrays.target_ids),
        "n_context": len(arrays.context_ids),
    }


def predicted_correlations_for_arm(
    *,
    decoder_module: ModuleType,
    arm: str,
    relationship: Mapping[str, Any],
    arrays: CellArrays,
    pair_ridge: float | None,
) -> np.ndarray:
    """Predicted target-context relationship matrix for one arm."""
    n_targets = int(relationship["n_targets"])
    n_context = int(relationship["n_context"])
    if arm in RAW_ARMS:
        return np.asarray(relationship["raw_d1"], dtype=np.float64)
    if pair_ridge is None:
        raise SpatialDecompositionError(f"{arm} requires a pair ridge")
    train, boolean, names = slice_arm(
        relationship["master_train"], relationship["master_boolean"], arm
    )
    target, _, _ = slice_arm(
        relationship["master_target"], relationship["master_boolean"], arm
    )
    labels = np.asarray(relationship["pair_labels"], dtype=np.float64)
    row_mask = (
        arrays.same_field_train_mask if arm.startswith("WF_")
        else np.ones(len(train), dtype=bool)
    )
    model = decoder_module.fit_pair_ridge(
        np.ascontiguousarray(train[row_mask]),
        np.ascontiguousarray(labels[row_mask]),
        alpha=float(pair_ridge),
        feature_names=names,
        boolean_mask=boolean,
    )
    predicted = decoder_module.predict_pair_ridge(model, target)
    return np.asarray(predicted, dtype=np.float64).reshape(n_targets, n_context)


def decode_arm(
    *,
    decoder_module: ModuleType,
    arm: str,
    predicted: np.ndarray,
    relationship: Mapping[str, Any],
    arrays: CellArrays,
    covariance_ridge: float,
    context_block: np.ndarray,
) -> np.ndarray:
    """Standardized decoder predictions on the supplied context block."""
    if not arm.startswith("WF_"):
        stacked, _diagnostics = decoder_module.decode_test_context(
            context_block,
            relationship["context_normalization"],
            predicted,
            relationship["context_correlation"],
            covariance_ridge=float(covariance_ridge),
        )
        return np.asarray(stacked, dtype=np.float64)

    # Within-field arms restrict each target to context neurons in its own field.
    n_samples = len(context_block)
    n_targets = int(relationship["n_targets"])
    output = np.zeros((n_samples, n_targets), dtype=np.float64)
    normalization = relationship["context_normalization"]
    full_correlation = np.asarray(relationship["context_correlation"], dtype=np.float64)
    standardized = decoder_module.apply_population_standardization(
        context_block, normalization
    )
    for position in range(n_targets):
        columns = np.asarray(arrays.per_target_field_context[position], dtype=np.int64)
        sub_correlation = full_correlation[np.ix_(columns, columns)]
        sub_predicted = np.asarray(predicted[position, columns], dtype=np.float64)[None, :]
        weights = decoder_module.covariance_decoder_weights(
            sub_predicted, sub_correlation, ridge=float(covariance_ridge)
        )
        gain = decoder_module.self_consistent_gain(
            sub_predicted, weights, sub_correlation, epsilon=NUMERICAL_EPSILON
        )
        output[:, position] = (standardized[:, columns] @ weights[0]) * float(gain[0])
    return output


def columnwise_pearson(true_values: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    true_centered = true_values - true_values.mean(axis=0, keepdims=True)
    predicted_centered = predicted - predicted.mean(axis=0, keepdims=True)
    numerator = np.sum(true_centered * predicted_centered, axis=0)
    denominator = np.sqrt(
        np.sum(true_centered * true_centered, axis=0)
        * np.sum(predicted_centered * predicted_centered, axis=0)
    )
    result = np.full(true_values.shape[1], np.nan, dtype=np.float64)
    valid = np.isfinite(denominator) & (denominator > NUMERICAL_EPSILON)
    result[valid] = numerator[valid] / denominator[valid]
    return np.clip(result, -1.0, 1.0)


def regenerate_pseudo_context_pairs(
    context_index: np.ndarray, scan: str, test_hash: str,
    outer_fold: int, pseudo_target_fold: int,
) -> tuple[np.ndarray, int]:
    """Deterministically sample context pairs for a pseudo-target fold."""
    indices = np.asarray(np.sort(context_index), dtype=np.int32)
    seed = stable_seed(
        PHASE8_SEED, scan, test_hash, outer_fold, pseudo_target_fold, "phase8d_pseudo_pairs"
    )
    left, right = np.triu_indices(len(indices), k=1)
    pairs = np.column_stack([indices[left], indices[right]]).astype(np.int32)
    if len(pairs) > MAX_TRAIN_PAIRS:
        rng = np.random.default_rng(seed)
        chosen = np.sort(rng.choice(len(pairs), size=MAX_TRAIN_PAIRS, replace=False))
        pairs = pairs[chosen]
    return pairs, seed


def verify_pseudo_pair_set(
    split: SplitInputs, pairs: np.ndarray, scan: str, rotation: int,
    outer_fold: int, direction: int,
) -> str:
    manifest = split.pair_set_manifest
    rows = manifest[
        (manifest["scan"] == scan)
        & (manifest["rotation_id"] == rotation)
        & (manifest["outer_neuron_fold"] == outer_fold)
        & (manifest["pair_set_role"] == "pseudo_context_context")
        & (manifest["validation_direction"] == float(direction))
    ]
    if len(rows) != 1:
        raise SpatialDecompositionError(
            f"pseudo pair set is not unique for {scan} rot{rotation} "
            f"fold{outer_fold} dir{direction}"
        )
    digest = sha256_bytes(np.ascontiguousarray(pairs, dtype="<i4").tobytes(order="C"))
    expected = str(rows["pair_indices_sha256"].iloc[0])
    if digest != expected:
        raise SpatialDecompositionError(
            f"pseudo pair hash mismatch {scan} rot{rotation} fold{outer_fold} "
            f"dir{direction}: {digest} != {expected}"
        )
    return digest


@dataclass(frozen=True)
class PseudoCell:
    arrays: CellArrays
    validation_target: np.ndarray
    pseudo_target_fold: int
    pseudo_context_fold: int


def build_pseudo_cell(
    *, split: SplitInputs, responses: np.ndarray, scan: str, rotation: int,
    orientation: int, outer_fold: int, budget: int, subset_id: str, direction: int,
) -> PseudoCell:
    """One pseudo cell for hyperparameter selection.

    Outer targets are excluded by construction and the terminal test movie is never
    read: the pseudo label lives on the separate validation movie.
    """
    axis_of = split.condition_axis[scan]
    test_hash, validation_hash, train_hashes = movie_roles_for_rotation(split, rotation)
    _cal_pool, pop_pool = pool_repeats(split, orientation)
    cal_repeats = subset_repeats(split, orientation, subset_id)
    if len(cal_repeats) != budget:
        raise SpatialDecompositionError("calibration subset size does not match the budget")

    neurons = scan_neurons(split, scan)
    context_folds = [f for f in OUTER_FOLDS if f != outer_fold]
    pseudo_target_fold = context_folds[direction]
    pseudo_context_fold = context_folds[1 - direction]

    fold_column = neurons["outer_target_fold"]
    target_index = np.sort(
        neurons[fold_column == pseudo_target_fold]["neuron_index"].to_numpy(np.int32))
    context_index = np.sort(
        neurons[fold_column == pseudo_context_fold]["neuron_index"].to_numpy(np.int32))
    outer_targets = set(
        neurons[fold_column == outer_fold]["neuron_index"].to_numpy(np.int32).tolist())
    if outer_targets & (set(target_index.tolist()) | set(context_index.tolist())):
        raise SpatialDecompositionError("an outer target leaked into a pseudo cell")
    if set(target_index.tolist()) & set(context_index.tolist()):
        raise SpatialDecompositionError("pseudo target and pseudo context overlap")

    xyz = neurons[["unit_x_um", "unit_y_um", "unit_z_um"]].to_numpy(np.float64)
    field = neurons["field"].to_numpy()
    area = neurons["brain_area"].to_numpy()
    row_of = {int(n): i for i, n in enumerate(neurons["neuron_index"].to_numpy())}

    # Unlicensed guard: selection may not read the terminal movie at all, for any neuron.
    guard = TerminalGuard(
        test_hash,
        frozenset(int(v) for v in neurons["neuron_index"].to_numpy()),
        licensed=False,
    )
    stack = lambda hashes, repeats, cols: stack_samples(
        responses, axis_of, hashes, repeats, cols, guard=guard)

    pairs, _seed = regenerate_pseudo_context_pairs(
        context_index, scan, test_hash, outer_fold, pseudo_target_fold)
    verify_pseudo_pair_set(split, pairs, scan, rotation, outer_fold, direction)
    local_of = {int(n): i for i, n in enumerate(context_index)}
    pairs_local = np.column_stack([
        np.asarray([local_of[int(v)] for v in pairs[:, 0]], dtype=np.int64),
        np.asarray([local_of[int(v)] for v in pairs[:, 1]], dtype=np.int64),
    ])

    field_by_neuron = {int(n): field[row_of[int(n)]] for n in neurons["neuron_index"].to_numpy()}
    same_field_train = np.asarray(
        [field_by_neuron[int(a)] == field_by_neuron[int(b)] for a, b in pairs], dtype=bool)
    if not same_field_train.any():
        raise SpatialDecompositionError("pseudo within-field pair training set is empty")

    per_target: dict[int, np.ndarray] = {}
    for position, neuron in enumerate(target_index):
        per_target[position] = np.flatnonzero(np.asarray(
            [field_by_neuron[int(c)] == field_by_neuron[int(neuron)] for c in context_index],
            dtype=bool)).astype(np.int64)

    ids = lambda arr: tuple(f"{scan}::{int(v)}" for v in arr)
    arrays = CellArrays(
        target_ids=ids(target_index),
        context_ids=ids(context_index),
        calibration_target=stack(train_hashes, cal_repeats, target_index),
        calibration_context=stack(train_hashes, cal_repeats, context_index),
        pair_label_context=stack(train_hashes, pop_pool, context_index),
        test_context=stack([validation_hash], pop_pool, context_index),
        train_pairs_global=pairs,
        train_pairs_local=pairs_local,
        target_context_pairs_global=target_context_pairs(target_index, context_index),
        same_field_train_mask=same_field_train,
        per_target_field_context=MappingProxyType(per_target),
        field_of_target=np.asarray([field_by_neuron[int(n)] for n in target_index]),
        xyz_um=xyz, field=field, brain_area=area,
        neuron_index_to_row=MappingProxyType(row_of),
    )
    return PseudoCell(
        arrays=arrays,
        validation_target=stack([validation_hash], pop_pool, target_index),
        pseudo_target_fold=pseudo_target_fold,
        pseudo_context_fold=pseudo_context_fold,
    )


def arm_grid(arm: str) -> tuple[tuple[float | None, float], ...]:
    """The specified hyperparameter grid for one arm."""
    if arm in RAW_ARMS:
        return tuple((None, c) for c in COVARIANCE_RIDGE_GRID)
    return tuple((p, c) for p in PAIR_RIDGE_GRID for c in COVARIANCE_RIDGE_GRID)


def select_hyperparameters(
    grid_means: Mapping[str, Mapping[tuple[float | None, float], float]],
) -> dict[str, dict[str, Any]]:
    """argmax with the specified tie-break: larger covariance ridge, then larger pair ridge."""
    chosen: dict[str, dict[str, Any]] = {}
    for arm in sorted(grid_means):
        grid = grid_means[arm]
        best_key: tuple[float | None, float] | None = None
        best_value: float | None = None
        rank = lambda k: (k[1], -1.0 if k[0] is None else k[0])
        for key in sorted(grid, key=lambda k: (-1.0 if k[0] is None else k[0], k[1])):
            value = float(grid[key])
            if not math.isfinite(value):
                raise SpatialDecompositionError(
                    f"non-finite pseudo-target validation score for {arm} at {key}")
            if best_value is None or value > best_value:
                best_key, best_value = key, value
            elif value == best_value and rank(key) > rank(best_key):
                best_key = key
        chosen[arm] = {
            "pair_ridge": best_key[0],
            "covariance_ridge": best_key[1],
            "mean_pseudo_validation_pearson": float(best_value),
        }
    return chosen
