from pathlib import Path
import json
import torch
from torch.utils.data import Dataset
from fsd_reality.geometry import ego_waypoints
from .cache import load_episode
from .splits import audit_cache_splits


class CachedEpisodeChunkDataset(Dataset):
    def __init__(self, root, split="train", chunk_size=8, jitter=True, seed=42):
        self.root = Path(root)
        self.split=split
        self.provenance=audit_cache_splits(root)
        self.paths = sorted((self.root/split).glob("*/episode.safetensors"))
        if not self.paths:
            raise ValueError(f"No episodes in {self.root/split}; collect data first")
        self.chunk_size,self.jitter,self.seed = chunk_size,jitter,seed
        self.episodes = [load_episode(path) for path in self.paths]
        self.metadata=[json.loads(path.with_name("metadata.json").read_text()) for path in self.paths]
        self.index = [(ep,start) for ep,e in enumerate(self.episodes)
                      for start in range(chunk_size,len(e["chase_latents"])-max(chunk_size,36),chunk_size)]
        if not self.index:
            raise ValueError("Episodes are too short for chunks plus 12 future waypoints")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        ep,start = self.index[index]
        e = self.episodes[ep]
        jitter = torch.randint(-1,2,(4,)) if self.jitter else torch.zeros(4,dtype=torch.long)
        ids = (start+torch.tensor([-1,1,3,5])+jitter).clamp(0,len(e["uve_tokens"])-1)
        stage_times = start+torch.tensor([0,2,4,6])
        fps=float(self.metadata[ep].get("fps",36))
        timestamps=e.get("timestamps",torch.arange(len(e["uve_tokens"]),dtype=torch.float64)/fps)
        return {"z_prev":e["chase_latents"][start-self.chunk_size:start].permute(1,0,2,3).float(),
                "z0":e["chase_latents"][start:start+self.chunk_size].permute(1,0,2,3).float(),
                "uve_tokens":e["uve_tokens"][ids].float(),
                "stage_ages":(timestamps[stage_times]-timestamps[ids]).float(),
                "waypoints":ego_waypoints(e["ego_states"].float(),start),
                "episode":ep,"start":start,"stage_frames":ids,"fps":fps,
                "source":self.metadata[ep].get("source","unknown"),
                "source_id":self.metadata[ep].get("source_id",str(self.paths[ep].parent))}
