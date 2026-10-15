"""Prepare portable response arrays from existing arrays or local DANDI NWB files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FILENAME = 'Y_clip_fluorescence_raw.npy'


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def verify(path, specification):
    array = np.load(path, mmap_mode='r', allow_pickle=False)
    if list(array.shape) != specification['shape'] or str(array.dtype) != specification['dtype']:
        raise ValueError(f'Response shape/dtype differs for {specification["scan"]}')
    if digest(path) != specification['sha256']:
        raise ValueError(f'Response checksum differs for {specification["scan"]}')


def from_nwb(nwb_path, units, trials, specification, output):
    """Replay the published extraction windows and original unit-column order.

    Windows use timestamps of the first retained field and their shortest common
    length. These saved indices preserve the original half-open extraction,
    including its use of the same window for every field.
    """
    import h5py
    result = np.empty(specification['shape'], dtype=specification['dtype'])
    with h5py.File(nwb_path, 'r') as handle:
        for field, group in units.groupby('field', sort=True):
            columns = group.neuron_index.to_numpy(dtype=int)
            roi = group.roi_index0.to_numpy(dtype=int)
            dataset = handle[f'/processing/ophys/Fluorescence/RoiResponseSeries{int(field)}/data']
            for trial in trials.itertuples():
                start, stop = int(trial.frame_i0), int(trial.frame_i1_exclusive)
                block = np.asarray(dataset[start:stop, :], dtype=np.float32)
                target = result[int(trial.condition_index), int(trial.repeat_index), :, :]
                target[:, columns] = block[:, roi]
    np.save(output, result, allow_pickle=False)
    verify(output, specification)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--replication-root', type=Path,
                        help='Existing SCAN/processed response arrays')
    inputs.add_argument('--nwb-root', type=Path,
                        help='Downloaded NWB files; preserves DANDI subdirectories or uses basenames')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scans', nargs='+', help='Default: all ten scans')
    args = parser.parse_args(argv)
    specs = json.loads((ROOT / 'configs/data/responses.json').read_text())
    scans = set(args.scans or [item['scan'] for item in specs])
    unknown = scans - {item['scan'] for item in specs}
    if unknown:
        parser.error(f'Unknown scans: {sorted(unknown)}')
    units = pd.read_csv(ROOT / 'configs/data/units.csv')
    trials = pd.read_csv(ROOT / 'configs/data/trials.csv')
    assets = pd.read_csv(ROOT / 'configs/data/assets.csv').set_index('scan')
    for specification in specs:
        scan = specification['scan']
        if scan not in scans:
            continue
        output = args.output / scan / 'processed' / FILENAME
        if output.exists():
            verify(output, specification)
            print(f'{scan}: existing array verified', flush=True)
            continue
        if args.replication_root:
            source = args.replication_root / scan / 'processed' / FILENAME
            verify(source, specification)
        else:
            relative = assets.loc[scan, 'asset_path']
            source = args.nwb_root / relative
            if not source.is_file():
                source = args.nwb_root / Path(relative).name
            if not source.is_file():
                raise FileNotFoundError(f'NWB missing: {relative}; see configs/data/assets.csv')
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix='.npy', delete=False) as stream:
            temporary = Path(stream.name)
        try:
            if args.replication_root:
                shutil.copyfile(source, temporary)
                verify(temporary, specification)
            else:
                from_nwb(source, units[units.scan == scan], trials[trials.scan == scan],
                         specification, temporary)
            if output.exists():
                raise FileExistsError(f'Output appeared during preparation: {output}')
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
        print(f'{scan}: prepared and checksum verified', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
