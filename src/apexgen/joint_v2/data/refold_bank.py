"""Validated train-only teacher bank for the shared Joint-v2 codesign runtime."""
import json
from pathlib import Path
import numpy as np
import torch
from apexgen.joint_v2.sampling.codesign_runtime import BASE_SCHEMA,tensors_digest,unpack_base
from apexgen.joint_v2.runtime.lineage import sha256_file
from apexgen.joint_v2.evaluation.generation_quality import generation_metrics
from apexgen.joint_v2.training.refold_reward import generated_record

SCHEMA='apexgen.joint_v2.refold_teacher_bank.v1'
AA='ARNDCQEGHILKMFPSTWYV'


def load_refold_bank(path,*,train_ids,validation_ids):
    train,valid=set(train_ids),set(validation_ids)
    if train&valid:raise ValueError('Training and validation IDs overlap')
    bank=json.loads(Path(path).read_text())
    if bank.get('schema')!=SCHEMA:raise ValueError('Unsupported teacher bank schema')
    hashes=bank['input_sha256']
    for p,h in hashes.items():
        if not Path(p).is_absolute() or not Path(p).is_file() or sha256_file(p)!=h:
            raise ValueError('Teacher bank input identity mismatch: '+p)
    for role in ['confirmation_panel','confirmation_metrics']:
        if bank[role] not in hashes:raise ValueError('Unbound '+role)
    panel=json.loads(Path(bank['confirmation_panel']).read_text())
    scores=json.loads(Path(bank['confirmation_metrics']).read_text())
    cases={c['name']:c for c in panel['cases']}
    if len(cases)!=len(panel['cases']):raise ValueError('Duplicate confirmation case')
    seeds=panel['prediction_seeds']
    if len(seeds)!=2 or len(set(seeds))!=2 or set(seeds)&set(bank['proposal_prediction_seeds']):
        raise ValueError('Confirmation seeds must be distinct and separate from proposal scoring')
    expected={(seed,sample) for seed in seeds for sample in range(2)}
    if not bank['teachers']:raise ValueError('Teacher bank is empty')
    ids=set(); assignments=set(); rows=[]
    for teacher in bank['teachers']:
        sid=teacher['sample_id'];tid=teacher['teacher_id']
        if sid not in train or sid in valid:raise ValueError('Teacher is outside the training split')
        if tid in ids:raise ValueError('Duplicate teacher identifier')
        ids.add(tid)
        base_path=teacher['base_path']
        if base_path not in hashes:raise ValueError('Teacher noise is not hash-bound')
        payload=torch.load(base_path,map_location='cpu',weights_only=True)
        if payload.get('schema')!=BASE_SCHEMA or payload['state_sha256']!=tensors_digest(payload['state'].items()):
            raise ValueError('Invalid stored teacher noise')
        key=(sid,payload['condition_sha256'],payload['state_sha256'])
        if key in assignments:raise ValueError('Conflicting or duplicate condition/noise assignment')
        assignments.add(key)
        sequence=teacher['sequence'];bb=np.asarray(teacher['generated_backbone'],dtype=np.float32)
        if len(sequence)<3 or any(a not in AA for a in sequence) or bb.shape!=(len(sequence),3,3) or not np.isfinite(bb).all():
            raise ValueError('Invalid teacher sequence/backbone')
        case=cases[teacher['confirmation_case']]
        if case['kind']!='generated' or case['sample_id']!=sid or case['sequence']!=sequence:
            raise ValueError('Teacher and confirmation case differ')
        rotation=np.asarray(teacher['condition_to_raw_rotation']);shift=np.asarray(teacher['condition_to_raw_translation'])
        if rotation.shape!=(3,3) or shift.shape!=(3,) or not np.isfinite(rotation).all() or not np.isfinite(shift).all():
            raise ValueError('Invalid coordinate mapping')
        if not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(rotation),1,atol=1e-5):
            raise ValueError('Coordinate mapping must be a proper rotation')
        if not np.allclose(bb[:,1]@rotation+shift,np.asarray(case['design_CA']),atol=1e-5,rtol=0):
            raise ValueError('Confirmation does not score the teacher design')
        for model in ['af3','boltz2']:
            group=[r for r in scores if r['name']==case['name'] and r['model']==model]
            if len(group)!=4 or {(r['seed'],r['sample']) for r in group}!=expected:
                raise ValueError('Incomplete or duplicate teacher confirmation')
            if any(r['sample_id']!=sid or r['path'] not in hashes for r in group):
                raise ValueError('Confirmation structures are not bound to the teacher')
            if sum(r['geometry_and_site_flag'] is True for r in group)<3:
                raise ValueError('Teacher fails repeated dual-predictor confirmation')
        rows.append(dict(teacher,generated_aatype=[AA.index(a) for a in sequence],base_payload=payload))
    return bank,rows


def bind_teacher_record(native_record,teacher,condition):
    if native_record['sample_id']!=teacher['sample_id']:raise ValueError('Wrong receptor record for teacher')
    if condition.layout[0]!=1:raise ValueError('Bind a teacher to one unpadded condition at a time')
    base=unpack_base(teacher['base_payload'],condition)
    record=generated_record(native_record,teacher)
    bb=torch.as_tensor(teacher['generated_backbone'],dtype=torch.float32,device=condition.pocket_translation.device)
    aa=torch.as_tensor(teacher['generated_aatype'],dtype=torch.long,device=bb.device)
    atoms=bb.new_zeros(len(bb),14,3);atoms[:,:3]=bb
    mask=torch.zeros(len(bb),14,dtype=torch.bool,device=bb.device);mask[:,:3]=True
    quality=generation_metrics(bb,atoms,mask,aa,condition,0)
    if not quality['generation_quality.geometry_and_core_site_proxy']:
        raise ValueError('Teacher fails generated backbone geometry/core checks')
    record['supervision_source']='train_only_confirmed_refold_teacher'
    return record,base
