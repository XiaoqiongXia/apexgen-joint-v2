"""Verify downloaded files, static caches, CSV rows and training collation."""
import csv
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / 'src'))

from apexgen.joint_v2.data.dataset import JointV2Dataset
from apexgen.joint_v2.data.static_features import read_static_features
from apexgen.joint_v2.data.training_collate import prepare_training_collator
from apexgen.joint_v2.runtime.lineage import joint_v2_dataset_identity, sha256_file


def verify():
    example = Path(__file__).resolve().parent
    checked_files = 0
    for line in (example / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        path = (example / name).resolve()
        if not path.is_relative_to(example) or sha256_file(path) != digest:
            raise ValueError(f'file checksum mismatch: {name}; check Git LFS download')
        checked_files += 1
    data = example / 'dataset'
    identity = joint_v2_dataset_identity(data)
    report = json.loads((data / 'static_features_upgrade.json').read_text())
    if report['status'] != 'complete' or report['dataset_identity'] != identity:
        raise ValueError('static feature upgrade identity mismatch')
    with (data / 'sample_inventory.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    dataset = JointV2Dataset(data, split='smoke')
    try:
        if len(dataset) != 8192 or len(rows) != len(dataset):
            raise ValueError('expected 8192 LMDB records and CSV rows')
        seen = set()
        for index, row in enumerate(rows):
            record = dataset[index]
            sample_id = record['sample_id']
            if (sample_id in seen or sample_id != row['sample_id']
                    or sample_id != row['tensor_key']
                    or record['peptide_length'] != int(row['target_length'])
                    or record['raw_file_sha256'] != row['source_npz_sha256']):
                raise ValueError(f'CSV/record mismatch at index {index}')
            if not Path(record['raw_path']).is_file():
                raise ValueError(f'missing relocated source at index {index}')
            if read_static_features(record) is None:
                raise ValueError(f'missing static features at index {index}')
            seen.add(sample_id)
        collator = prepare_training_collator(dataset, verified_identity=identity)
        if collator.report['mode'] != 'light':
            raise ValueError('expected lightweight training mode; disable debug override')
        print(json.dumps(dict(status='passed', samples=len(dataset), files=checked_files,
            static_caches_checked=len(seen), dataset_identity=identity,
            input_validation=collator.report), indent=2))
    finally:
        dataset.close()


if __name__ == '__main__':
    verify()
