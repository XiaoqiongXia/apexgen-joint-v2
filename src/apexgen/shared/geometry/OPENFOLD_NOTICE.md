# OpenFold numerical reference notice

_Pinned upstream identity and attribution for ApexGen v0 geometry cross-validation_

---

## 🔗 Reference

ApexGen v0 pins the official [OpenFold repository](https://github.com/aqlaboratory/openfold)
at commit
[`be2ec1841f16c966c65ae0e7599ebbadc725757d`](https://github.com/aqlaboratory/openfold/commit/be2ec1841f16c966c65ae0e7599ebbadc725757d).

The numerical cross-check targets are:

- `openfold/np/residue_constants.py`
- `openfold/utils/rigid_utils.py`

## 📋 License and use

OpenFold source code is distributed under the
[Apache License 2.0](https://github.com/aqlaboratory/openfold/blob/be2ec1841f16c966c65ae0e7599ebbadc725757d/LICENSE).
ApexGen does not require the full OpenFold runtime. `joint-v1` includes the minimal adapted
numerical table subset in `_openfold_joint_constants.py`; the corresponding upstream Apache-2.0
license is retained verbatim in `LICENSE-OPENFOLD`. Its public wrapper is
`joint_residue_constants.py`. Future modifications must preserve this attribution, the pinned
commit, and numerical equivalence for the exported atom14/chi tables.
