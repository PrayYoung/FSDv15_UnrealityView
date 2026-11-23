"""Return compact evidence; keep native data and feature caches on the remote host."""
import argparse
from pathlib import Path
import shutil
from unrealityview.data.cache import atomic_json
from unrealityview.data.dataset import CachedEpisodeChunkDataset

p=argparse.ArgumentParser(description=__doc__);p.add_argument("--data",required=True);p.add_argument("--output",required=True);a=p.parse_args()
root=Path(a.data);out=Path(a.output)/"data_evidence";out.mkdir(parents=True,exist_ok=True)
for split in ("train","val","test"):
    data=CachedEpisodeChunkDataset(root,split,jitter=False)
    for path,meta in zip(data.paths,data.metadata):
        name=path.parent.name
        shutil.copyfile(path.with_name("raw_preview.png"),out/f"{split}_{name}.png")
    atomic_json(out/f"{split}_inventory.json",{"episodes":data.metadata,"chunks":len(data)})
shutil.copyfile(root/"manifest.json",out/"manifest.json")
print("EXPORTED_COMPACT_REAL_DATA_EVIDENCE",flush=True)
