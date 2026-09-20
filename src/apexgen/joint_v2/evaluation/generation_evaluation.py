"""Native-reference-free summaries of archived, fixed-sequence geometry samples.

The historical proximity proxy uses receptor context, not the specified pocket
core. Neither it nor the geometry proxy establishes binding or full validity.
"""

from itertools import combinations

import numpy as np


def ca_pair_distances(left, right):
    """Return common-frame and proper-rotation-aligned RMSD in Angstroms."""
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3 or len(a) < 3:
        raise ValueError("Expected matching unpadded CA arrays of shape (N>=3, 3)")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite generated coordinates")
    common = np.sqrt(np.mean(np.sum((a - b) ** 2, axis=-1)))
    ac, bc = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(ac.T @ bc)
    correction = np.diag([1.0, 1.0, np.linalg.det(u @ vt)])
    aligned = ac @ (u @ correction @ vt)
    internal = np.sqrt(np.mean(np.sum((aligned - bc) ** 2, axis=-1)))
    return float(common), float(internal)


def geometry_record(row):
    """Read an explicit allowlist: no native error or native coordinates."""
    names = {
        'cn_mae': 'cn_absolute_mae_angstrom',
        'ca_spacing_mae': 'ca_spacing_mae_angstrom',
        'context_clash_fraction': 'peptide_atom_clash_fraction_2A',
        'context_proximity_fraction': 'contact_residue_fraction_5A',
    }
    result = {name: float(row[source]) for name, source in names.items()}
    if not all(np.isfinite(v) and v >= 0 for v in result.values()):
        raise ValueError("Invalid geometry metric")
    if any(result[k] > 1 for k in ['context_clash_fraction', 'context_proximity_fraction']):
        raise ValueError("Fraction outside [0, 1]")
    if not isinstance(row['nonlocal_backbone_clash'], bool):
        raise ValueError("Expected Boolean nonlocal clash")
    result['nonlocal_clash'] = row['nonlocal_backbone_clash']
    result['geometry_proxy'] = (
        result['cn_mae'] <= .15 and result['context_clash_fraction'] <= .05
        and not result['nonlocal_clash']
    )
    result['context_proximity_proxy'] = result['context_proximity_fraction'] >= .5
    result['historical_combined_proxy'] = (
        result['geometry_proxy'] and result['context_proximity_proxy']
    )
    if result['historical_combined_proxy'] != row['backbone_valid_proxy']:
        raise ValueError("Historical proxy cannot be reproduced")
    return result


def summarize_target(rows):
    """Average four paired bases within one target; retain empty valid-pair sets."""
    if len(rows) != 4 or sorted(r['base_index'] for r in rows) != [0, 1, 2, 3]:
        raise ValueError("Expected exactly four distinct bases")
    rows = sorted(rows, key=lambda r: r['base_index'])
    records = [geometry_record(r) for r in rows]
    result = {k: float(np.mean([r[k] for r in records])) for k in records[0]}
    result['geometry_count'] = sum(r['geometry_proxy'] for r in records)
    result['combined_count'] = sum(r['historical_combined_proxy'] for r in records)
    result['any_geometry'] = result['geometry_count'] > 0
    result['any_combined'] = result['combined_count'] > 0
    pairs = []
    for i, j in combinations(range(4), 2):
        common, internal = ca_pair_distances(rows[i]['predicted_CA'], rows[j]['predicted_CA'])
        pairs.append(dict(base_left=i, base_right=j, common_frame_rmsd=common,
                          internal_shape_rmsd=internal,
                          both_geometry=records[i]['geometry_proxy'] and records[j]['geometry_proxy']))
    for prefix, selected in [('all', pairs), ('geometry', [p for p in pairs if p['both_geometry']])]:
        result[f'{prefix}_pair_count'] = len(selected)
        for metric in ['common_frame_rmsd', 'internal_shape_rmsd']:
            result[f'{prefix}_{metric}'] = float(np.mean([p[metric] for p in selected])) if selected else None
    return result, pairs
