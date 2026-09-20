"""Verify this downloaded subset and collate boundary records from each shard."""
from pathlib import Path
import json
import sys

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / 'src'))
from apexgen.joint_v2.data.dataset import JointV2Dataset
from apexgen.joint_v2.data.batch import collate_joint_v2_records
from apexgen.joint_v2.runtime.lineage import joint_v2_dataset_identity

if __name__ == '__main__':
    data = Path(__file__).resolve().parent / 'dataset'
    identity = joint_v2_dataset_identity(data)
    dataset = JointV2Dataset(data, split='smoke')
    try:
        assert len(dataset) == 24576
        for index in (0, 8191, 8192, 16383, 16384, 24575):
            record = dataset[index]
            assert Path(record['raw_path']).is_file()
            collate_joint_v2_records([record]).condition.validate_model_input()
        print(json.dumps(dict(samples=len(dataset), status='passed', dataset_identity=identity)))
    finally:
        dataset.close()
