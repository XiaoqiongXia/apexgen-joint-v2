#!/usr/bin/env python
"""Export a small runnable repository from a committed tree or the Git index.

No datasets outside examples/, checkpoints, local paths, or experiment history
are copied. Exporting never changes the source repository's branches/history.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[2]
FILES = {
    ".gitignore", ".gitattributes", "pyproject.toml", "scripts/__init__.py",
    "requirements/portable-tested-py312.txt",
    "scripts/project_tmp_env.sh", "scripts/release/export_simplex_core.py",
    "scripts/data/build_boltz_interface_manifest.py",
    "scripts/data/prepare_joint_v2_boltzgen.py",
    "scripts/data/prepare_joint_v2_boltz_npz.py",
    "scripts/data/prepare_joint_v2_boltz_chain_pairs.py",
    "configs/joint_v2/experiments/sequence_structure_tiny_encoder_bottleneck_v1.yaml",
    "docs/portable_simplex_training.md", "docs/boltzgen_training_pipeline.md",
    "docs/repository_language.md",
    "docs/boltz_interface_fragment_training.md", "docs/boltz_interface_manifest.md",
    "docs/joint_v2_boltz_npz_adapter.md",
    "tests/__init__.py", "tests/joint_v2/__init__.py", "tests/joint_v2/conftest.py",
    "tests/joint_v2/test_boltz_npz.py", "tests/joint_v2/test_boltzgen_npz.py",
    "tests/joint_v2/test_boltz_chain_pairs.py", "tests/joint_v2/test_boltz_interfaces.py",
    "tests/joint_v2/test_portable_simplex.py",
}
PREFIXES = ("src/apexgen/", "configs/joint_v2/portable/", "examples/boltzgen19/")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def check_english_text(name, payload):
    """Reject untranslated CJK prose while retaining scientific Unicode symbols."""
    if Path(name).suffix in {".npz", ".parquet", ".mdb"}:
        return
    try:
        content = payload.decode("utf-8")
    except UnicodeDecodeError:
        return
    if re.search(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U000323af]", name + content):
        raise ValueError(f"English-only release contains CJK text: {name}")


def export(output, *, index=False):
    paths = git("ls-files", "-z") if index else git("ls-tree", "-r", "--name-only", "-z", "HEAD")
    available = {p.decode() for p in paths.split(b"\0") if p}
    selected = {p for p in available if p in FILES or p.startswith(PREFIXES)}
    # Keep small referenced Boltz reports as historical evidence.
    selected.update(p for p in available if p.startswith("reports/boltz") and p.endswith(".md"))
    required = FILES - {p for p in FILES if p.endswith("/__init__.py")}
    if required - selected:
        raise ValueError(f"missing source files in selected Git tree: {sorted(required - selected)}")
    payloads = {name: git("show", (":" if index else "HEAD:") + name) for name in sorted(selected)}
    for name, payload in payloads.items():
        check_english_text(name, payload)
    output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name, payload in payloads.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        hashes[name] = hashlib.sha256(payload).hexdigest()
    readme = (output / "docs/portable_simplex_training.md").read_bytes()
    (output / "README.md").write_bytes(readme)
    hashes["README.md"] = hashlib.sha256(readme).hexdigest()
    instructions = (output / "docs/repository_language.md").read_bytes()
    (output / "AGENTS.md").write_bytes(instructions)
    hashes["AGENTS.md"] = hashlib.sha256(instructions).hexdigest()
    snapshot = dict(
        schema="apexgen.simplex_core_export.v1", source_commit=git("rev-parse", "HEAD").decode().strip(),
        source="index" if index else "HEAD", files=hashes,
    )
    (output / "SOURCE_SNAPSHOT.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    print(json.dumps(dict(files=len(hashes), bytes=sum((output / p).stat().st_size for p in hashes))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--index", action="store_true", help="validate staged changes before committing")
    args = parser.parse_args()
    export(args.output.resolve(), index=args.index)
