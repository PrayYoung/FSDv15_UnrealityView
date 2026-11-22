"""Single-producer / multiple-reader CPython ring with immutable publication.

Not a claim of portable C++ atomics or interprocess lock freedom. Publication is
one Python reference assignment after cloning/completing the tensor payload.
Readers hold that snapshot reference even when its physical slot is reused.
"""
from dataclasses import dataclass
import threading
import time
import torch


@dataclass(frozen=True)
class UVESnapshot:
    tokens: torch.Tensor
    timestamp: float
    sequence_id: int
    frame_id: int


class UVERingBuffer:
    def __init__(self, num_slots=9):
        if num_slots < 2:
            raise ValueError("At least two ring slots are required")
        self.slots = [None]*num_slots
        self._latest_idx = -1
        self._latest = None
        self._sequence = 0
        self._writer = None

    def publish(self, tokens, ts=None, frame_id=None):
        writer = threading.get_ident()
        if self._writer is not None and writer != self._writer:
            raise RuntimeError("UVERingBuffer supports exactly one producer thread")
        self._writer = writer
        payload = tokens.detach().clone()
        if payload.device.type == "cuda":
            torch.cuda.current_stream(payload.device).synchronize()
        elif payload.device.type == "mps":
            torch.mps.synchronize()
        seq = self._sequence
        snapshot = UVESnapshot(payload,time.monotonic() if ts is None else float(ts),seq,
                               seq if frame_id is None else int(frame_id))
        slot = seq % len(self.slots)
        self.slots[slot] = snapshot  # fully written before publication
        self._latest_idx = slot
        self._latest = snapshot      # atomic reference assignment under CPython GIL
        self._sequence += 1
        return snapshot

    def read_latest_nonblocking(self):
        """Do not mutate returned tensor. The producer never mutates published storage."""
        return self._latest


class RealityViewStreamingEngine:
    def __init__(self, model, ring):
        self.model,self.ring = model,ring
        self.last_telemetry = {}

    @torch.no_grad()
    def sample_chunk(self, previous, num_steps=1, interleaving=True, noise=None,
                     clock=time.monotonic, stage_hook=None):
        initial = self.ring.read_latest_nonblocking()
        if initial is None:
            raise RuntimeError("No completed UVE frame available")
        records = []

        def provider(stage):
            if stage_hook is not None:
                stage_hook(stage)
            newest = self.ring.read_latest_nonblocking() if interleaving else initial
            now = clock()
            age = max(0.,now-newest.timestamp)
            gain = max(0.,newest.timestamp-initial.timestamp)
            records.append({"stage":stage,"sequence_id":newest.sequence_id,"frame_id":newest.frame_id,
                            "age_ms":age*1000,"freshness_gain_ms":gain*1000})
            return newest.tokens,previous.new_full((len(previous),),age)

        out = self.model.sample_chunk(previous,initial.tokens,num_steps,uve_provider=provider,noise=noise)
        self.last_telemetry = {"stages":records,"initial_sequence_id":initial.sequence_id,
                               "freshness_gain_ms":max((r["freshness_gain_ms"] for r in records),default=0.)}
        return out
