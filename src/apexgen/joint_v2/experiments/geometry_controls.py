"""Exploratory fixed-sequence geometry controls, with distinct checkpoint identity.

Anchoring changes the reference frame of every proposal, not the existing 1-t gate.
Chain supervision leaves the baseline decoder computation unchanged.
"""
import math
import torch
from torch import Tensor, nn
from apexgen.shared.geometry.residue_constants import BACKBONE_GEOMETRY
from apexgen.joint_v2.contracts.contract import JointEndpointPrediction, UnifiedComplexCondition, UnifiedComplexEncoding
from apexgen.joint_v2.experiments.fixed_sequence_scale import FixedSequenceScaleModel
from apexgen.joint_v2.runtime.lineage import canonical_sha256
from apexgen.joint_v2.model.variants.neighbor_attention import peptide_neighbor_attention
from apexgen.joint_v2.geometry.reconstruction import reconstruct_backbone
from apexgen.joint_v2.contracts.state import JointFlowState, AMINO_ACID_TYPES, BACKBONE_ANGLE_SLOTS, center_sequence_logits
from apexgen.joint_v2.model.structure_module import JointV2StructureModule, Rigid
from apexgen.joint_v2.model.task_factorization import task_losses as baseline_losses, task_objective_weights as baseline_weights

CONTROLS=('baseline','anchored','chain')
CONTRACT=dict(schema='apexgen.joint_v2.geometry_controls.v1',task='G_s',
    anchored='Each block composes the existing gated quaternion/translation proposal with original input frames, while IPA observes current proposed frames; unchanged head, time gate and losses',
    chain='Original accumulation; add final CN length and two bridge-angle cosine squared errors, averaged over valid peptide bonds then samples, weight1',
    chain_constants=dict(c_n=BACKBONE_GEOMETRY.c_n,ca_c_n=BACKBONE_GEOMETRY.ca_c_n,c_n_ca=BACKBONE_GEOMETRY.c_n_ca),
    checkpoint='Exploratory explicit control contract; never a formal or generic baseline-compatible checkpoint')
CONTRACT_SHA256=canonical_sha256(CONTRACT)

class AnchoredGeometryModule(JointV2StructureModule):
    def __init__(self, baseline):
        # Reuse the just-initialized children without random draws or renamed parameters.
        nn.Module.__init__(self)
        if list(baseline.named_parameters(recurse=False)) or list(baseline.named_buffers(recurse=False)):
            raise ValueError('Unexpected top-level structure-module tensors')
        for name, child in baseline.named_children():
            self.add_module(name,child)
        for name in ['sequence_output_parameterization','sequence_attention_topology','blocks','translation_scale','stop_rotation_gradient']:
            setattr(self,name,getattr(baseline,name))
        self.train(baseline.training)

    def forward(
        self,
        state: JointFlowState,
        time: Tensor,
        condition: UnifiedComplexCondition,
        encoding: UnifiedComplexEncoding,
        *,
        return_intermediates: bool = True,
        task: str = "J",
        trace: list | None = None,
        initialize_sequence_latent: bool = True,
    ) -> JointEndpointPrediction:
        if task not in {"J", "S", "G_s", "G_0"}:
            raise ValueError(f"unknown refinement task {task}")
        if state.layout != condition.layout or encoding.single.shape[:2] != condition.layout:
            raise ValueError("decoder state, condition and encoding layouts differ")
        if time.shape != (condition.layout[0],):
            raise ValueError("time must be [batch]")
        single = self.layer_norm_s(encoding.single)
        pair = self.layer_norm_z(encoding.pair)
        initial_single = single
        single = self.linear_in(single)
        time_embedding = self.time_conditioner(time)
        gamma, beta = self.time_film(time_embedding).chunk(2, dim=-1)
        block_update_gate = (1.0 - time.float())[:, None, None]
        peptide = condition.peptide_mask[..., None]
        fixed_state = JointFlowState(
            translation=torch.where(
                condition.peptide_mask[..., None],
                state.translation,
                condition.pocket_translation,
            ),
            rotation=torch.where(
                condition.peptide_mask[..., None, None],
                state.rotation,
                condition.pocket_rotation,
            ),
            sequence_logits=torch.where(
                peptide,
                state.sequence_logits,
                torch.zeros_like(state.sequence_logits),
            ),
        )
        rigid = Rigid.from_state(fixed_state, self.translation_scale)
        input_rigid = rigid
        sequence_single = (
            torch.zeros_like(single)
            if task == "G_0" or not initialize_sequence_latent
            else self.sequence_projection(self.sequence_norm(fixed_state.sequence_logits))
        )
        sequence_single = torch.where(peptide, sequence_single, 0.0)
        translations: list[Tensor] = []
        rotations: list[Tensor] = []

        final_state = fixed_state
        attention_mask = (
            peptide_neighbor_attention(condition, self.ipa.no_heads)
            if self.sequence_attention_topology == "neighbors"
            else None
        )
        for block_index in range(self.blocks):
            combined = single + sequence_single
            conditioned = combined * (1.0 + peptide * gamma[:, None]) + peptide * beta[:, None]
            if attention_mask is None:
                single = single + self.ipa(conditioned, pair, rigid, condition.residue_mask)
            else:
                single = single + self.ipa(
                    conditioned, pair, rigid, condition.residue_mask, attention_mask=attention_mask
                )
            single = self.layer_norm_ipa(self.ipa_dropout(single))
            single = self.transition(single)
            update = (
                torch.zeros(*single.shape[:2], 6, device=single.device, dtype=single.dtype)
                if task == "S"
                else self.backbone_update(single) * block_update_gate
            )
            with torch.autocast(device_type=single.device.type, enabled=False):
                update = update.float()
                if task != "S":
                    proposed = input_rigid.compose_q_update_vec(update)
                    rigid = rigid.masked(proposed, condition.peptide_mask)
                raw_export = rigid.export(self.translation_scale, fixed_state.sequence_logits)
                exported = JointFlowState(
                    translation=torch.where(
                        peptide,
                        raw_export.translation if task != "S" else fixed_state.translation,
                        fixed_state.translation,
                    ),
                    rotation=torch.where(
                        peptide[..., None],
                        raw_export.rotation if task != "S" else fixed_state.rotation,
                        fixed_state.rotation,
                    ),
                    sequence_logits=torch.where(
                        peptide, raw_export.sequence_logits, fixed_state.sequence_logits
                    ),
                )
                final_state = exported
            if return_intermediates:
                translations.append(exported.translation)
                rotations.append(exported.rotation)
            if task != "G_0":
                sequence_single = self.sequence_latent_transition(
                    sequence_single,
                    single,
                    condition.peptide_mask,
                )
            if trace is not None:
                entry = {
                    "iteration": block_index,
                    "single": single,
                    "sequence_latent": sequence_single,
                    "update": update,
                    "translation": exported.translation,
                    "rotation": exported.rotation,
                }
                for value in entry.values():
                    if isinstance(value, Tensor) and value.requires_grad:
                        value.retain_grad()
                trace.append(entry)
            if self.stop_rotation_gradient and block_index + 1 < self.blocks:
                rigid = rigid.stop_rotation_gradient()

        if task in {"J", "S"}:
            sequence_single = self.final_sequence_resnet(single, initial_single, sequence_single)
        with torch.autocast(device_type=single.device.type, enabled=False):
            sequence_delta = (
                self.sequence_update(sequence_single.float()).float()
                if task in {"J", "S"}
                else torch.zeros_like(fixed_state.sequence_logits)
            )
            proposed_sequence = (
                center_sequence_logits(
                    fixed_state.sequence_logits + sequence_delta * block_update_gate
                    if self.sequence_output_parameterization == "residual"
                    else sequence_delta
                )
                if task in {"J", "S"}
                else fixed_state.sequence_logits
            )
            sequence = torch.where(peptide, proposed_sequence, 0.0)
            raw_export = rigid.export(self.translation_scale, sequence)
            final_state = JointFlowState(
                translation=torch.where(
                    peptide,
                    raw_export.translation if task != "S" else fixed_state.translation,
                    fixed_state.translation,
                ),
                rotation=torch.where(
                    peptide[..., None],
                    raw_export.rotation if task != "S" else fixed_state.rotation,
                    fixed_state.rotation,
                ),
                sequence_logits=torch.where(
                    peptide, raw_export.sequence_logits, fixed_state.sequence_logits
                ),
            )

        unnormalized_angles, angles = self.angle_resnet(single, initial_single)
        peptide_angles = condition.peptide_mask[..., None, None]
        unnormalized_angles = torch.where(peptide_angles, unnormalized_angles, 0.0)
        angles = torch.where(peptide_angles, angles, 0.0)
        unnormalized_backbone = unnormalized_angles[..., :BACKBONE_ANGLE_SLOTS, :]
        backbone_angles = angles[..., :BACKBONE_ANGLE_SLOTS, :]
        unnormalized_sidechain = unnormalized_angles[..., BACKBONE_ANGLE_SLOTS:, :]
        sidechain_angles = angles[..., BACKBONE_ANGLE_SLOTS:, :]
        return JointEndpointPrediction(
            translation=final_state.translation,
            rotation=final_state.rotation,
            sequence_logits=final_state.sequence_logits,
            block_translation=(torch.stack(translations, dim=1) if translations else None),
            block_rotation=(torch.stack(rotations, dim=1) if rotations else None),
            block_sequence_logits=None,
            unnormalized_backbone_angles=unnormalized_backbone,
            backbone_angles_sin_cos=backbone_angles,
            unnormalized_sidechain_angles=unnormalized_sidechain,
            sidechain_angles_sin_cos=sidechain_angles,
        )


class GeometryControlModel(FixedSequenceScaleModel):
    def __init__(self, config, control, code_seed=20260906):
        if control not in CONTROLS:raise ValueError('Unknown geometry control')
        super().__init__(config,code_seed)
        self.control=control
        if control=='anchored':
            self.decoder.structure_module=AnchoredGeometryModule(self.decoder.structure_module)


def build_fit_model(config,task,protocol):
    if task!='G_s':raise ValueError('Geometry controls require fixed sequence G_s')
    return GeometryControlModel(config,protocol['control'],protocol['position_code_seed'])


def chain_continuity_terms(backbone,condition):
    """Chemistry-only final CN/angle loss; no native peptide geometry is read."""
    with torch.autocast(device_type=backbone.device.type,enabled=False):
        xyz=backbone.float()
        ca0,c0,n1,ca1=xyz[:,:-1,1],xyz[:,:-1,2],xyz[:,1:,0],xyz[:,1:,1]
        bond=n1-c0
        def cosine(u,v):
            return (u*v).sum(-1)/(u.norm(dim=-1).clamp_min(1e-6)*v.norm(dim=-1).clamp_min(1e-6))
        mask=(condition.peptide_mask[:,:-1]&condition.peptide_mask[:,1:]
              &(condition.chain_index[:,:-1]==condition.chain_index[:,1:])
              &(condition.sequence_index[:,1:]-condition.sequence_index[:,:-1]==1))
        def mean(x):return torch.where(mask,x,0).sum(-1)/mask.sum(-1).clamp_min(1)
        terms=dict(chain_cn=mean((bond.norm(dim=-1)-BACKBONE_GEOMETRY.c_n).square()),
                   chain_ca_c_n=mean((cosine(ca0-c0,bond)-math.cos(BACKBONE_GEOMETRY.ca_c_n)).square()),
                   chain_c_n_ca=mean((cosine(-bond,ca1-n1)-math.cos(BACKBONE_GEOMETRY.c_n_ca)).square()))
        terms['chain_continuity']=sum(terms.values())/3
        return terms


def control_weights(task,control):
    if control not in CONTROLS or task!='G_s':raise ValueError('Invalid geometry task/control')
    values=baseline_weights(task)
    if control=='chain':values['chain_continuity']=1.0
    return values


def control_losses(prediction,batch,task,control):
    values=baseline_losses(prediction,batch,task)
    if control=='chain':
        xyz=reconstruct_backbone(JointFlowState(prediction.translation.float(),prediction.rotation.float(),prediction.sequence_logits.float()))
        values.update(chain_continuity_terms(xyz,batch.condition))
        values['total']=values['total']+values['chain_continuity']
    elif control not in CONTROLS:raise ValueError('Unknown geometry control')
    return values
