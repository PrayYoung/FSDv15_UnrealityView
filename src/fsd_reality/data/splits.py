"""Frozen scene-level splits and provenance; never split adjacent frame windows."""
import hashlib
import json
from pathlib import Path


def stable_split(dataset,sequence,seed=42):
    value=int(hashlib.sha256(f"{seed}/{dataset}/{sequence}".encode()).hexdigest()[:16],16)/2**64
    return "train" if value<.8 else "val" if value<.9 else "test"


def validate_split_manifest(manifest):
    seen={}
    for entry in manifest["sequences"]:
        if entry["dataset"] not in ("waymo","pandaset") or entry["split"] not in ("train","val","test"):
            raise ValueError("Unsupported dataset or split")
        key=f"{entry['dataset']}/{entry['sequence']}"
        if key in seen:raise ValueError(f"Sequence appears more than once (possible leakage): {key}")
        seen[key]=entry["split"]
        if entry["dataset"]=="waymo" and entry.get("official_partition")=="validation" and entry["split"]!="test":
            raise ValueError("Reserved Waymo official validation belongs only to final test")
    return manifest


def audit_cache_splits(root):
    seen={};sources={}
    for split in ("train","val","test"):
        for path in sorted((Path(root)/split).glob("*/metadata.json")):
            meta=json.loads(path.read_text())
            # Legacy/synthetic caches have no real-source identity.
            identity=meta.get("source_id")
            if identity:
                if identity in seen and seen[identity]!=split:raise ValueError(f"Source leakage: {identity}")
                seen[identity]=split
                sources[str(path.parent)]=identity
    return seen


def assert_unseen_test(provenance,test_ids):
    if provenance is None:raise ValueError("Checkpoint lacks training provenance; cannot certify held-out testing")
    overlap=set(provenance.get("train_source_ids",[]))&set(test_ids)
    if overlap:raise ValueError(f"Test sequences overlap training: {sorted(overlap)}")
