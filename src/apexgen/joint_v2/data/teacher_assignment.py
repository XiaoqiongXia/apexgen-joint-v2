"""Validate deterministic rollout supervision within each initialized model."""


def assigned_teachers(teachers, parent_seed):
    selected=sorted((t for t in teachers if t.get('parent_seed')==parent_seed), key=lambda t:t['teacher_id'])
    if not selected:
        raise ValueError('No historical teacher pairs for this parent')
    ids=set(); bases=set()
    for t in selected:
        key=(t['sample_id'],t['base_sha256'])
        if t['teacher_id'] in ids:
            raise ValueError('Duplicate teacher identifier')
        if key in bases:
            raise ValueError('Duplicate condition/base assignment within one model')
        ids.add(t['teacher_id']); bases.add(key)
    return selected
