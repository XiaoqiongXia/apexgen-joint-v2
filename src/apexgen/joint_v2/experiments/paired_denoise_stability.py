"""Single-factor continuation controls for paired local geometry restoration."""
from apexgen.joint_v2.runtime.lineage import canonical_sha256

ARMS=('lr_decay','full_rotation','baseline_replay')
CONTRACT=dict(schema='apexgen.joint_v2.paired_denoise_stability.v1',
    parent='128-target paired single-task raw step4000, exact Adam/EMA/RNG and global target sampler restoration',
    lr_decay='Only linear LR decay:1e-4 at4000 to1e-5 at8000; original rotation stop',
    full_rotation='Only disable rotation-state gradient stop; original constant1e-4 LR',
    reference='Already completed parent4000->8000; baseline_replay is preflight only',
    scope='Exploratory continuation at fixed sequence/low noise; no claim about from-scratch interventions or full-base generation')
CONTRACT_SHA256=canonical_sha256(CONTRACT)


def learning_rate(arm,global_step,base=1e-4,start=4000,end=8000,floor_fraction=.1):
    if arm not in ARMS or not start<=global_step<=end:raise ValueError((arm,global_step))
    if arm=='lr_decay':return base*(1-(1-floor_fraction)*(global_step-start)/(end-start))
    return base


def configure(model,arm):
    if arm not in ARMS:raise ValueError(arm)
    model.decoder.structure_module.stop_rotation_gradient=arm!='full_rotation'
