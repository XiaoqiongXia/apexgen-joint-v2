"""Explicit train-teacher support and confirmation gates, without native recovery."""
import math
import numpy as np


def raw_refold_failures(row, *, plddt_min=None):
    failures = []
    for key, limit in [('CN_MAE', .15), ('peptide_heavy_clash_fraction_2A', .05)]:
        value = float(row[key])
        if not math.isfinite(value) or value < 0 or value > limit:
            failures.append(key)
    if row['nonlocal_backbone_clash'] is not False:
        failures.append('nonlocal_backbone_clash')
    if row['pocket_retained_flag'] is not True:
        failures.append('core_site')
    if plddt_min is not None:
        value = float(row['peptide_plddt_0_100'])
        if not math.isfinite(value) or not plddt_min <= value <= 100:
            failures.append('plddt')
    return failures


def repeat_group(rows, *, name, sample_id, model, seeds):
    if len(seeds) != 2 or len(set(seeds)) != 2:
        raise ValueError('Expected two distinct predictor seeds')
    group = [r for r in rows if r['name'] == name and r['model'] == model]
    expected = {(seed, sample) for seed in seeds for sample in range(2)}
    if len(group) != 4 or {(r['seed'], r['sample']) for r in group} != expected:
        raise ValueError('Incomplete or duplicate predictor repeats: ' + name)
    if any(r['sample_id'] != sample_id or r['kind'] != 'generated' for r in group):
        raise ValueError('Predictor repeat belongs to a different candidate')
    return group


def supported_medoid(rows, *, radius=2., minimum_support=3):
    """Return one actual observed backbone with cross-seed support; never average."""
    if radius <= 0 or minimum_support < 3:
        raise ValueError('Invalid support gate')
    if len({(r['seed'], r['sample']) for r in rows}) != len(rows):
        raise ValueError('Duplicate structures cannot increase support')
    choices = []
    for row in rows:
        ca = np.asarray(row['generated_backbone'], dtype=float)[:, 1]
        if not np.isfinite(ca).all():
            raise ValueError('Nonfinite teacher backbone')
        distances = [float(np.sqrt(np.mean(np.sum((ca-np.asarray(other['generated_backbone'])[:, 1])**2, axis=-1))))
                     for other in rows]
        cluster = [(other,d) for other,d in zip(rows, distances) if d <= radius]
        if len(cluster) < minimum_support or len({other['seed'] for other,_ in cluster}) < 2:
            continue
        mean = float(np.mean([d for _,d in cluster]))
        score = (-len(cluster), mean, -row['plddt'], row['seed'], row['sample'])
        choices.append((score, dict(row, support=[dict(seed=r['seed'], sample=r['sample']) for r,_ in cluster],
                                   support_mean_rmsd=mean)))
    return min(choices,key=lambda x:x[0])[1] if choices else None


def diverse_target_selection(rows, *, maximum=4):
    """One best candidate per sequence first, then remaining candidates by rank."""
    if maximum < 1:
        raise ValueError('Positive teacher budget required')
    if len({r['sample_id'] for r in rows}) > 1:
        raise ValueError('Selection must be within a single target')
    if len({r['name'] for r in rows}) != len(rows):
        raise ValueError('Duplicate candidate')
    ordered = sorted(rows,key=lambda r:(-len(r['support']),r['support_mean_rmsd'],-r['plddt'],r['name']))
    first,rest,seen = [],[],set()
    for row in ordered:
        if row['sequence'] in seen:
            rest.append(row)
        else:
            first.append(row)
            seen.add(row['sequence'])
    return (first+rest)[:maximum]


def confirmed_models(rows, *, case, seeds, minimum_passes=3):
    """A passing old pose/site flag alone is insufficient for teacher admission."""
    counts = {}
    for model in ['af3', 'boltz2']:
        group = repeat_group(rows,name=case['name'],sample_id=case['sample_id'],model=model,seeds=seeds)
        counts[model] = sum(not raw_refold_failures(r) and
            math.isfinite(float(r['receptor_aligned_peptide_CA_RMSD'])) and
            0 <= float(r['receptor_aligned_peptide_CA_RMSD']) <= 2 and
            r['geometry_and_site_flag'] is True for r in group)
    return all(n >= minimum_passes for n in counts.values()), counts
