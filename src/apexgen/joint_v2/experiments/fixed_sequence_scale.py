"""Fixed-sequence geometry baseline for independent receptor coverage scaling."""
from torch import nn
from apexgen.joint_v2.model.task_factorization import TaskFactorizationModel
from apexgen.joint_v2.model.variants.joint_position_fit import position_codes
from apexgen.joint_v2.model.structure_module import OpenFoldLinear
from apexgen.joint_v2.runtime.lineage import canonical_sha256

CONTRACT = dict(
    schema='apexgen.joint_v2.fixed_sequence_scale.v1',
    task='G_s: native sequence fixed at every refinement and solver step; no native peptide geometry observation',
    architecture='Original task-factorization G_s plus label-independent random rank encoder position input; global IPA',
    data='Nested32/128 distinct receptor components; frozen14 validation; same sequence/pocket exclusions',
    geometry='Unchanged independent-frame output, geometry losses and 20-step solver; isolate data coverage first',
    budgets='8000 steps per size/seed; compare equal compute at8000 and equal250exposures at32/2000 versus128/8000',
    compatibility='Exploratory only; no checkpoint compatibility or formal training claim',
)
CONTRACT_SHA256=canonical_sha256(CONTRACT)

class FixedSequenceScaleModel(TaskFactorizationModel):
    def __init__(self,config,code_seed=20260906):
        super().__init__(config,'G_s')
        self.code_seed=code_seed
        width=config['architecture'].get('encoder_single_dim',config['architecture']['single_dim'])
        self.encoder.position_norm=nn.LayerNorm(20)
        self.encoder.position_projection=OpenFoldLinear(20,width)

    def encode_complex(self,observation):
        if observation.task!='G_s':raise ValueError('fixed sequence geometry requires G_s observations')
        z=position_codes(observation.pocket.peptide_mask,'random',self.code_seed)
        injected=self.encoder.position_projection(self.encoder.position_norm(z))
        return self.encoder(observation.pocket,peptide_single=injected)

def build_fit_model(config,task,protocol):
    if task!='G_s':raise ValueError('fixed sequence scale requires G_s')
    return FixedSequenceScaleModel(config,protocol['position_code_seed'])
