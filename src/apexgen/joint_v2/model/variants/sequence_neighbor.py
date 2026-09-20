"""Matched likelihood-model control for directional IPA message routing."""

from apexgen.joint_v2.runtime.lineage import canonical_sha256
from apexgen.joint_v2.model.variants.sequence_likelihood import LikelihoodModel, CONTRACT_SHA256 as PARENT_SHA256

CONTRACT = dict(
    schema="apexgen.joint_v2.neighbor_routing.v1",
    parent_contract=PARENT_SHA256,
    endpoint="Known Gaussian likelihood plus learned correction, unchanged",
    intervention="IPA heads 0/1/2 read self/left/right peptide sites; other heads global",
    boundaries="Same chain, sequence index delta -1/+1; missing neighbor uses self; no cyclic wrap",
    pocket="Pocket queries remain global in every head",
    parameters="No added parameters; same tensor initialization; routing changes effective gradients",
    compatibility="Exploratory only; topology and source manifest must match",
)
CONTRACT_SHA256 = canonical_sha256(CONTRACT)


class NeighborModel(LikelihoodModel):
    def __init__(self, topology):
        if topology not in {"global", "neighbors"}:
            raise ValueError("unknown sequence topology")
        super().__init__("likelihood")
        self.topology = topology
        self.decoder.structure_module.sequence_attention_topology = topology
