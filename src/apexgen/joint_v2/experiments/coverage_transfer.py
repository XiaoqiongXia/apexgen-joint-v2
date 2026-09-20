"""Exploratory data-scale contract and split-aware loading; architecture unchanged."""
from apexgen.joint_v2.runtime.lineage import canonical_sha256
from apexgen.joint_v2.data.dataset import JointV2Dataset
from apexgen.joint_v2.data.batch import collate_joint_v2_records
from apexgen.joint_v2.model.task_factorization import aligned_batch

CONTRACT = dict(schema='apexgen.joint_v2.coverage_transfer.v1',
    architecture='LocalDenoiseModel with original DynamicPairStructureModule, full rotation gradients, 8 shared blocks',
    initialization='Archived random step0 per seed; zero dynamic pair projection; fresh AdamW and EMA; no learned checkpoint',
    data='Nested32/128 independent receptor components, old development14, additional metadata-selected holdout7 from original train',
    comparison='Equal8192steps and equal128 direction-pair visits/target:32/2048 versus128/8192',
    task='Native sequence, clean and joint local denoising, unchanged clean proposal penalty; full-base rollout is outside local training support',
    holdout='No access to additional7 in optimization or monitoring; predeclared checkpoints evaluated only after all training is complete',
    scope='Exploratory local transfer; no blind project-wide holdout or formal generation claim')
CONTRACT_SHA256 = canonical_sha256(CONTRACT)


def load_pools(panel, data, device, names):
    """Honor source split, independently of the experimental pool name."""
    datasets = {}
    lookup = {}
    records, singles, batches = {}, {}, {}
    try:
        for name in names:
            records[name] = []
            for item in panel[name]:
                split = item['split']
                if split not in datasets:
                    datasets[split] = JointV2Dataset(data['pocket_root'], data['target_root'], split)
                    lookup[split] = {r['sample_id']:i for i,r in enumerate(datasets[split].rows)}
                row = datasets[split][lookup[split][item['sample_id']]]
                records[name].append(row)
                singles[item['sample_id']] = aligned_batch(collate_joint_v2_records([row]).to(device))
            batches[name] = [aligned_batch(collate_joint_v2_records(records[name][i:i+4]).to(device))
                             for i in range(0,len(records[name]),4)]
    finally:
        for dataset in datasets.values():
            dataset.close()
    return records, singles, batches
