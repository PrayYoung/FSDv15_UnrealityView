from pathlib import Path
import json
import torch
from torch.utils.data import Dataset
from unrealityview.geometry import ego_waypoints
from .cache import load_episode,episode_grid
from .splits import audit_cache_splits


def stage_context(episode,start,fps=36,jitter=None):
    count=len(episode["uve_tokens"])
    times=episode.get("timestamps",torch.arange(count,dtype=torch.float64)/fps)
    complete=episode["camera_timestamps"].max(1).values if "camera_timestamps" in episode else times
    jitter=torch.zeros(4,dtype=torch.long) if jitter is None else jitter
    anchors=start+torch.tensor([0,2,4,6])
    requested=(start+torch.tensor([-1,1,3,5])+jitter).clamp(0,count-1)
    # A six-camera snapshot cannot be used until its LAST camera was captured.
    # This also prevents +1 jitter from exposing cameras recorded after an anchor.
    latest=torch.searchsorted(complete.contiguous(),times[anchors],right=True)-1
    if (latest<0).any():raise ValueError("No completed camera snapshot at stage anchor")
    ids=torch.minimum(requested,latest)
    return ids,(times[anchors]-complete[ids]).float()


class CachedEpisodeChunkDataset(Dataset):
    def __init__(self, root, split="train", chunk_size=8, jitter=True, seed=42, stride: int | None=None):
        self.root = Path(root)
        self.split=split
        self.provenance=audit_cache_splits(root)
        self.paths = sorted((self.root/split).glob("*/episode.safetensors"))
        if not self.paths:
            raise ValueError(f"No episodes in {self.root/split}; collect data first")
        self.stride=(2 if split=="train" else chunk_size) if stride is None else stride
        if type(self.stride) is not int or self.stride<1:raise ValueError("stride must be a positive integer")
        self.chunk_size,self.jitter,self.seed = chunk_size,jitter,seed
        self.episodes = [load_episode(path) for path in self.paths]
        self.metadata=[json.loads(path.with_name("metadata.json").read_text()) for path in self.paths]
        grids={episode_grid(e,m) for e,m in zip(self.episodes,self.metadata)}
        if len(grids)!=1:raise ValueError('Mixed UVE grids in dataset')
        self.uve_grid=next(iter(grids))
        manifest=self.root/'manifest.json'
        if manifest.exists():
            declared=json.loads(manifest.read_text()).get('uve_grid')
            if declared is not None and tuple(declared)!=self.uve_grid:raise ValueError('Manifest UVE grid mismatch')
        self.index = [(ep,start) for ep,e in enumerate(self.episodes)
                      for start in range(chunk_size,len(e["chase_latents"])-max(chunk_size,36),self.stride)]
        if not self.index:
            raise ValueError("Episodes are too short for chunks plus 12 future waypoints")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        ep,start = self.index[index]
        e = self.episodes[ep]
        jitter = torch.randint(-1,2,(4,)) if self.jitter else torch.zeros(4,dtype=torch.long)
        fps=float(self.metadata[ep].get("fps",36))
        ids,ages=stage_context(e,start,fps,jitter)
        return {"z_prev":e["chase_latents"][start-self.chunk_size:start].permute(1,0,2,3).float(),
                "z0":e["chase_latents"][start:start+self.chunk_size].permute(1,0,2,3).float(),
                "uve_tokens":e["uve_tokens"][ids].float(),
                "policy_tokens":e["uve_tokens"][start].float(),
                "stage_ages":ages,
                "waypoints":ego_waypoints(e["ego_states"].float(),start),
                "episode":ep,"start":start,"stage_frames":ids,"fps":fps,
                "source":self.metadata[ep].get("source","unknown"),
                "source_id":self.metadata[ep].get("source_id",self.paths[ep].parent.name)}
