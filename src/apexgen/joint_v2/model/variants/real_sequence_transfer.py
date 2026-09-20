"""Exploratory real-pocket transfer of sequence observation and neighbor routing."""

from dataclasses import replace
import torch
from torch import nn

from apexgen.joint_v2.model.variants.joint_position_fit import JointPositionFitModel, JOINT_POSITION_CONTRACT_SHA256
from apexgen.joint_v2.runtime.lineage import canonical_sha256
from apexgen.joint_v2.model.variants.sequence_neighbor import CONTRACT_SHA256 as NEIGHBOR_SHA256
from apexgen.joint_v2.model.variants.sequence_posterior import SequenceInput, SCALE
from apexgen.joint_v2.model.variants.sequence_likelihood import observation_logits
from apexgen.joint_v2.contracts.state import center_sequence_logits
from apexgen.joint_v2.sampling.clocks import sequence_time, validate_sequence_time_power

CONTRACT = dict(
    schema='apexgen.joint_v2.real_sequence_transfer.v3',
    parent_joint=JOINT_POSITION_CONTRACT_SHA256,
    parent_neighbor=NEIGHBOR_SHA256,
    input='normalized plus zero-initialized raw sequence branch; both arms',
    endpoint='known Gaussian emission plus 20*h/SCALE; centered posterior mean',
    endpoint_modes='likelihood (default) OR direct centered unconstrained continuous endpoint; explicit protocol sequence_endpoint_mode',
    sequence_clock='s=t**sequence_time_power; shared by path, likelihood and sampling; default 1',
    topology='global OR self/left/right peptide heads, other heads and pocket queries global',
    training='real J joint geometry and sampled sequence endpoint MSE; no Markov teacher, no sequence CE',
    compatibility='exploratory only; new parameters and endpoint interpretation require new checkpoint',
)
CONTRACT_SHA256=canonical_sha256(CONTRACT)


def sequence_output(raw, current, time, *, mode, power):
    """Shared training/sampling output; direct scores are readout logits, not a posterior."""
    with torch.autocast(device_type=raw.device.type, enabled=False):
        if mode == 'direct':
            endpoint = center_sequence_logits(raw.float())
            return endpoint, endpoint
        if mode != 'likelihood':
            raise ValueError('Unknown sequence endpoint mode')
        scores = observation_logits(current, sequence_time(time, power)) + 20/SCALE*raw.float()
        return center_sequence_logits(SCALE*(scores.softmax(-1)-.05)), scores


class RealSequenceTransferModel(JointPositionFitModel):
    def __init__(self, config, *, topology, position_mode, code_seed, sequence_time_power=1.0,
                 sequence_endpoint_mode='likelihood'):
        if topology not in {'global','neighbors'}:
            raise ValueError('unknown real sequence transfer topology')
        super().__init__(config,position_mode=position_mode,code_seed=code_seed)
        self.topology=topology
        self.sequence_time_power=validate_sequence_time_power(sequence_time_power)
        if sequence_endpoint_mode not in {'likelihood','direct'}:
            raise ValueError('Unknown sequence endpoint mode')
        self.sequence_endpoint_mode=sequence_endpoint_mode
        m=self.decoder.structure_module
        m.sequence_attention_topology=topology
        m.sequence_output_parameterization='direct'
        # Append after original encoder position-code modules: preserve shared initialization.
        m.sequence_projection=SequenceInput(m.sequence_norm,m.sequence_projection,True)
        m.sequence_norm=nn.Identity()

    def decode(self,state,time,observation,encoding,*,trace=None,return_intermediates=True):
        pred=super().decode(state,time,observation,encoding,trace=trace,return_intermediates=return_intermediates)
        current=observation.clamp(state)
        with torch.autocast(device_type=state.sequence_logits.device.type,enabled=False):
            sequence,_=sequence_output(pred.sequence_logits,current.sequence_logits,time,
                mode=self.sequence_endpoint_mode,power=self.sequence_time_power)
            sequence=torch.where(observation.pocket.peptide_mask[...,None],sequence,0.0)
        return replace(pred,sequence_logits=sequence)


def build_real_transfer(config,task,protocol):
    if task!='J':
        raise ValueError('real sequence transfer is a J experiment')
    return RealSequenceTransferModel(config,topology=protocol['sequence_topology'],
                                    position_mode=protocol['joint_position_mode'],
                                    code_seed=protocol['position_code_seed'],
                                    sequence_time_power=protocol.get('sequence_time_power',1.0),
                                    sequence_endpoint_mode=protocol.get('sequence_endpoint_mode','likelihood'))
