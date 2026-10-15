"""Run matched spatial estimators from the released splits and prepared responses."""
from __future__ import annotations

import os
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[variable] = '1'

import argparse
import itertools
import json
import hashlib
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import pandas as pd
from spatial_priors import estimator as model
from spatial_priors.pipeline import select_scope, predict_cell, score_cell, aggregate_cells


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True,
                   help='Folder containing SCAN/processed/Y_clip_fluorescence_raw.npy')
    p.add_argument('--split-root', type=Path, default=ROOT / 'configs/splits')
    p.add_argument('--output', type=Path, required=True, help='A new output directory')
    p.add_argument('--scans', nargs='+', help='Default: all ten scans')
    p.add_argument('--rotations', nargs='+', type=int, choices=range(6), default=list(range(6)))
    p.add_argument('--orientations', nargs='+', type=int, choices=(0, 1), default=[0, 1])
    p.add_argument('--folds', nargs='+', type=int, choices=(0, 1, 2), default=[0, 1, 2])
    p.add_argument('--budgets', nargs='+', type=int, choices=model.BUDGETS, default=list(model.BUDGETS))
    p.add_argument('--arms', nargs='+', choices=model.ARMS + ('D1_SHRUNK',),
                   default=['D1', 'D1_PAIR_CALIBRATED', 'D1_DXY'])
    p.add_argument('--policy', choices=['selected', 'fixed'], default='selected')
    p.add_argument('--save-predictions', action='store_true')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.output.exists():
        raise SystemExit('Choose a new output directory; existing results are not overwritten.')
    split = model.load_split_inputs(args.split_root)
    scans = args.scans or sorted(split.neuron_folds.scan.unique())
    unknown = set(scans) - set(split.neuron_folds.scan)
    if unknown:
        raise SystemExit(f'Unknown scans: {sorted(unknown)}')
    specifications = {item['scan']: item for item in json.loads(
        (ROOT / 'configs/data/responses.json').read_text())}
    for scan in scans:
        array = model.load_scan_responses(args.data_root, scan)
        expected = specifications[scan]
        if list(array.shape) != expected['shape'] or str(array.dtype) != expected['dtype']:
            raise SystemExit(f'Response shape/dtype differs for {scan}; run prepare_data.py.')
        path = args.data_root / scan / 'processed/Y_clip_fluorescence_raw.npy'
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != expected['sha256']:
            raise SystemExit(f'Response checksum differs for {scan}; run prepare_data.py.')
    args.output.mkdir(parents=True)
    settings_dir = args.output / 'selection'
    settings_dir.mkdir()
    configuration = {'scans': scans, 'rotations': args.rotations,
        'orientations': args.orientations, 'folds': args.folds, 'budgets': args.budgets,
        'arms': args.arms, 'policy': args.policy, 'complete': False}
    (args.output / 'run.json').write_text(json.dumps(configuration, indent=2) + '\n')
    rows = []
    start = time.monotonic()
    for scan in scans:
        responses = model.load_scan_responses(args.data_root, scan)
        for rotation, orientation, fold, budget in itertools.product(
                args.rotations, args.orientations, args.folds, args.budgets):
            scope_name = f'{scan}_r{rotation}_o{orientation}_f{fold}_b{budget}'
            if args.policy == 'selected':
                settings, grid = select_scope(split, responses, scan=scan,
                    rotation=rotation, orientation=orientation, outer_fold=fold,
                    budget=budget, arms=args.arms)
                grid_json = {arm: {f'{key[0]}|{key[1]}': value for key, value in values.items()}
                             for arm, values in grid.items()}
            else:
                settings = {arm: {'pair_ridge': None if arm in model.RAW_ARMS or arm == 'D1_SHRUNK' else 5.0,
                                  'covariance_ridge': 30.0} for arm in args.arms}
                if 'D1_SHRUNK' in settings:
                    settings['D1_SHRUNK']['rank_label'] = '64'
                grid_json = {}
            (settings_dir / f'{scope_name}.json').write_text(
                json.dumps({'selected': settings, 'validation_grid': grid_json}, indent=2) + '\n')
            for subset in model.subset_ids_for(split, orientation, budget):
                identity = model.CellIdentity(scan, rotation, orientation, fold, budget, subset)
                predictions, target_ids = predict_cell(split, responses, identity, settings)
                if args.save_predictions:
                    folder = args.output / 'predictions'
                    folder.mkdir(exist_ok=True)
                    np.savez_compressed(folder / f'{scope_name}_{subset}.npz',
                                        target_ids=np.asarray(target_ids), **predictions)
                scores = score_cell(split, responses, identity, predictions, target_ids)
                for row in scores:
                    row.update(policy_arm=args.policy, **settings[row['candidate']])
                rows.extend(scores)
            pd.DataFrame(rows).to_csv(args.output / 'cell_scores.csv', index=False)
            print(f'{scope_name}: complete ({time.monotonic() - start:.1f} seconds)', flush=True)
    cells = pd.DataFrame(rows)
    for name, frame in aggregate_cells(cells).items():
        frame.to_csv(args.output / f'{name}.csv', index=False)
    configuration['complete'] = True
    configuration['cells'] = len(cells)
    configuration['elapsed_seconds'] = time.monotonic() - start
    (args.output / 'run.json').write_text(json.dumps(configuration, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
