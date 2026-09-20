"""Fail-closed integrity verification for a separately frozen Joint-v2 run.

This verifies provenance and runtime inputs. It does not establish scientific
readiness or turn an exploratory checkpoint into a formal model.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

SCHEMA='apexgen.joint_v2.formal_run.v1'


class FormalLockError(RuntimeError):
    pass


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def verify_formal_lock(manifest_path, *, worktree):
    root=Path(worktree).resolve();mp=Path(manifest_path).resolve()
    def require(test, message):
        if not test:raise FormalLockError(message)
    def git(*args):
        p=subprocess.run(['git','-C',str(root),*args],capture_output=True,text=True)
        require(p.returncode==0,'git verification failed: '+' '.join(args))
        return p.stdout.strip()
    m=json.loads(mp.read_text())
    require(m.get('schema')==SCHEMA and m.get('formal') is True,'not a formal Joint-v2 manifest')
    require(Path(m['worktree']).resolve()==root,'worktree path mismatch')
    require(Path(git('rev-parse','--show-toplevel')).resolve()==root,'repository root mismatch')
    require(git('branch','--show-current')==m['branch'],'branch mismatch')
    require(git('rev-parse','HEAD')==m['commit'],'commit mismatch')
    require(not git('status','--porcelain','--untracked-files=all'),'dirty worktree')
    tag=m['tag'];require(tag==m['run_name'],'tag and run name differ')
    require(git('cat-file','-t','refs/tags/'+tag)=='tag','formal tag must be annotated')
    require(git('rev-parse','refs/tags/'+tag+'^{commit}')==m['commit'],'tag commit mismatch')
    contents=git('cat-file','-p','refs/tags/'+tag)
    require(('manifest-sha256: '+digest(mp)) in contents.splitlines(),'manifest not bound by annotated tag')
    roots=m['artifact_roots'];require(isinstance(roots,list) and bool(roots),'missing artifact roots')
    resolved=[]
    for relative in roots:
        p=Path(relative)
        require(bool(p.parts) and not p.is_absolute() and '..' not in p.parts and p.parts[0] in {'artifacts','runs'},'invalid artifact root')
        path=(root/p).resolve();require(path.is_relative_to(root),'artifact root escapes worktree')
        probe=subprocess.run(['git','-C',str(root),'check-ignore','-q',str(p/'formal_lock_probe')])
        require(probe.returncode==0,'artifact root is not ignored')
        resolved.append(path)
    require(any(mp.is_relative_to(p) for p in resolved),'manifest must reside in declared ignored artifact root')
    inputs=m['input_sha256'];require(isinstance(inputs,dict) and bool(inputs),'missing bound inputs')
    for role in ['config','data_manifest','validation_panel','readiness_evidence']:
        p=Path(m[role]);require(p.is_absolute() and str(p) in inputs,role+' not hash-bound')
    init=m.get('initialization_checkpoint')
    require(init is None or init in inputs,'initialization checkpoint not hash-bound')
    for p,h in inputs.items():
        require(Path(p).is_absolute() and Path(p).is_file(),'missing input: '+p)
        require(digest(p)==h,'input hash mismatch: '+p)
    require(m['python_version']==platform.python_version(),'Python version mismatch')
    deps=m['dependencies'];require(isinstance(deps,dict) and bool(deps),'missing dependency pins')
    for package,version in deps.items():
        try:actual=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:raise FormalLockError('missing dependency: '+package)
        require(actual==version,'dependency mismatch: '+package)
    return dict(passed=True,run_name=m['run_name'],commit=m['commit'],tag=tag,worktree=str(root),
                manifest_sha256=digest(mp),verified_inputs=len(inputs),artifact_roots=[str(p) for p in resolved],
                scientific_readiness_requires_separate_review=True)
