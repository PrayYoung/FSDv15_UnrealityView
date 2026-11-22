"""Loopback-only FastAPI cockpit. Launch: uv run python -m app.server"""
from collections import OrderedDict
import base64
from io import BytesIO
import json
import os
from pathlib import Path
import threading
import time
from typing import Annotated,Literal
from uuid import UUID
import numpy as np
from PIL import Image
import torch
from fastapi import FastAPI,HTTPException,Request
from fastapi.responses import FileResponse,JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,ConfigDict,Field
from starlette.middleware.trustedhost import TrustedHostMiddleware
from fsd_reality.checkpoint import load_checkpoint
from fsd_reality.data.cache import resolve_safe_subpath,load_episode
from fsd_reality.device import select_device,synchronize,autocast_for
from fsd_reality.geometry import ground_raycast,project_waypoints,ego_waypoints
from fsd_reality.models.pretrained_taesd import PretrainedTAESD
from fsd_reality.runtime.ring_buffer_engine import UVERingBuffer,RealityViewStreamingEngine
from .demo import demo_frame,demo_waypoints

APP_DIR=Path(__file__).resolve().parent
DATA_ROOT=Path(os.environ.get("REALITY_DATA","outputs/dataset")).resolve()
CHECKPOINT=Path(os.environ.get("REALITY_CHECKPOINT","outputs/train/checkpoint")).resolve()
app=FastAPI(title="Mini-RealityView",docs_url=None,redoc_url=None)
app.add_middleware(TrustedHostMiddleware,allowed_hosts=["127.0.0.1","localhost","testserver"])
app.mount("/static",StaticFiles(directory=APP_DIR/"static"),name="static")
_lock=threading.Lock()
_models=None
_episode_cache=OrderedDict()
_sessions=OrderedDict()


class TapGoal(BaseModel):
    model_config=ConfigDict(extra="forbid",allow_inf_nan=False,strict=True)
    u:float=Field(ge=0,le=320)
    v:float=Field(ge=0,le=192)


class RenderRequest(BaseModel):
    model_config=ConfigDict(extra="forbid",allow_inf_nan=False,strict=True)
    episode:str=Field(default="val/episode_0050",min_length=1,max_length=120)
    chunk:int=Field(default=1,ge=1,le=10000)
    nfe:Literal[1,2,4]=1
    source:Literal["model","ground_truth","demo"]="demo"
    rollout:Literal["teacher_forced","closed_loop"]="teacher_forced"
    interleaving:bool=True
    tiled:bool=False
    goal:TapGoal|None=None
    session_id:str=Field(pattern=r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
    seed:int=Field(default=42,ge=0,le=2**31-1)


@app.middleware("http")
async def local_security(request:Request,call_next):
    origin=request.headers.get("origin")
    if origin and origin not in ("http://127.0.0.1:8088","http://localhost:8088"):
        return JSONResponse({"detail":"Origin not allowed"},status_code=403)
    response=await call_next(request)
    response.headers["X-Content-Type-Options"]="nosniff"
    response.headers["Referrer-Policy"]="no-referrer"
    response.headers["Content-Security-Policy"]="default-src 'self'; img-src 'self' data:; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'"
    return response


@app.get("/")
def index():
    return FileResponse(APP_DIR/"index.html")


@app.get("/api/health")
def health():
    return {"status":"ok","device":str(select_device()),"checkpoint_available":(CHECKPOINT/"model.safetensors").is_file(),
            "dataset_available":any(DATA_ROOT.glob("*/*/episode.safetensors")),
            "purpose":"Independent research prototype; no vehicle control"}


@app.get("/api/episodes")
def episodes():
    rows=[]
    for path in sorted(DATA_ROOT.glob("*/*/episode.safetensors")):
        try:
            metadata=json.loads(path.with_name("metadata.json").read_text())
            rows.append({"id":str(path.parent.relative_to(DATA_ROOT)),"frames":metadata["frames"],
                         "source":metadata.get("source","unknown")})
        except (OSError,ValueError,KeyError):
            continue
    return {"episodes":rows}


@app.get("/api/assets/{path:path}")
def asset(path:str):
    try: target=resolve_safe_subpath(DATA_ROOT,path)
    except ValueError as e: raise HTTPException(400,str(e))
    if target.suffix.lower() not in (".jpg",".jpeg",".png",".gif") or not target.is_file():
        raise HTTPException(404,"Image not found")
    return FileResponse(target)


def image_url(image):
    buf=BytesIO();image.save(buf,format="JPEG",quality=88)
    return "data:image/jpeg;base64,"+base64.b64encode(buf.getvalue()).decode("ascii")


def video_urls(video):
    array=((video[0].detach().float().cpu().clamp(-1,1)+1)*127.5).round().to(torch.uint8).permute(1,2,3,0).numpy()
    return [image_url(Image.fromarray(frame)) for frame in array]


def get_episode(path):
    key=str(path)
    if key not in _episode_cache:
        _episode_cache[key]=load_episode(path/"episode.safetensors")
        while len(_episode_cache)>2: _episode_cache.popitem(last=False)
    _episode_cache.move_to_end(key)
    return _episode_cache[key]


def get_models():
    global _models
    if _models is None:
        if not (CHECKPOINT/"model.safetensors").is_file():
            raise HTTPException(409,"No trained checkpoint. Run training or select ground truth / procedural preview.")
        device=select_device()
        model,policy,config=load_checkpoint(CHECKPOINT,device)
        decoder=PretrainedTAESD().to(device).eval()
        _models=(model,policy,decoder,device,config)
    return _models


@app.post("/api/render")
@torch.no_grad()
def render(req:RenderRequest):
    try: directory=resolve_safe_subpath(DATA_ROOT,req.episode)
    except ValueError as e: raise HTTPException(400,str(e))
    goal=None if req.goal is None else ground_raycast(req.goal.u,req.goal.v)
    if req.goal is not None and goal is None:
        raise HTTPException(422,"Tap a visible road point below the horizon (0–100m ahead)")
    start=req.chunk*8
    cameras=("front_left","front","front_right","rear_left","rear","rear_right")
    if req.source=="demo":
        waypoints=demo_waypoints(goal)
        return {"frames":[image_url(demo_frame(start+i)) for i in range(8)],
                "cameras":[image_url(demo_frame(start,c)) for c in range(6)],
                "waypoints":waypoints,"projected_waypoints":project_waypoints(waypoints),"goal_xy":goal,
                "source":"procedural preview","trained":False,"frame_start":start,"fps":36,
                "telemetry":{"dit_ms":None,"decode_ms":None,"policy_ms":None,"effective_fps":None,
                             "psnr_db":None,"sram_mib":None,"uve_stage_frames":[],"stream_mode":"illustration"}}
    if not (directory/"episode.safetensors").is_file():
        raise HTTPException(404,"Episode cache missing; collect simulator data first")
    # Serialize model access and each user's closed-loop state. No browser-side
    # hidden state is trusted for model tensors or paths.
    with _lock:
        episode=get_episode(directory)
        if start+8>len(episode["chase_latents"]): raise HTTPException(422,"Chunk exceeds episode length")
        relative=str(directory.relative_to(DATA_ROOT))
        camera_urls=[f"/api/assets/{relative}/raw/{c}/{start:05d}.jpg" for c in cameras]
        if req.source=="ground_truth":
            waypoints=ego_waypoints(episode["ego_states"].float(),start).tolist()
            return {"frames":[f"/api/assets/{relative}/raw/chase/{start+i:05d}.jpg" for i in range(8)],
                    "cameras":camera_urls,"waypoints":waypoints,"projected_waypoints":project_waypoints(waypoints),
                    "goal_xy":goal,"source":"ground truth","trained":False,"frame_start":start,"fps":36,
                    "telemetry":{"dit_ms":None,"decode_ms":None,"policy_ms":None,"effective_fps":None,
                                 "psnr_db":None,"sram_mib":None,"uve_stage_frames":[],"stream_mode":"cached ground truth"}}
        model,policy,decoder,device,config=get_models()
        previous=episode["chase_latents"][start-8:start].permute(1,0,2,3)[None].float().to(device)
        sid=str(req.session_id)
        identity=(req.episode,req.nfe,req.seed,req.interleaving,req.tiled)
        old=_sessions.get(sid)
        if req.rollout=="closed_loop" and req.chunk>1:
            if old is None or old[0]!=identity or old[1]!=req.chunk-1:
                raise HTTPException(409,"Closed-loop playback must begin at chunk 1 and advance sequentially. Reset playback.")
            previous=old[2].to(device)
        ring=UVERingBuffer()
        first=start-1
        ring.publish(episode["uve_tokens"][first][None].float().to(device),ts=first/36,frame_id=first)
        engine=RealityViewStreamingEngine(model,ring)
        stage_current=[start/36]
        def stage_hook(stage):
            frame=min(start+stage*2-1,len(episode["uve_tokens"])-1)
            stage_current[0]=(start+stage*2)/36
            ring.publish(episode["uve_tokens"][frame][None].float().to(device),ts=frame/36,frame_id=frame)
        generator=torch.Generator(device=device).manual_seed(req.seed+req.chunk)
        noise=torch.randn(previous.shape,device=device,generator=generator)
        initial_tokens=episode["uve_tokens"][first][None].float().to(device)
        synchronize(device);begin=time.perf_counter()
        with autocast_for(device):
            points=policy(initial_tokens,None if goal is None else torch.tensor([goal],device=device))
        synchronize(device);policy_ms=(time.perf_counter()-begin)*1000
        begin=time.perf_counter()
        with autocast_for(device):
            latent=engine.sample_chunk(previous,req.nfe,req.interleaving,noise=noise,
                         clock=lambda:stage_current[0],stage_hook=stage_hook if req.interleaving else None)
        synchronize(device);dit_ms=(time.perf_counter()-begin)*1000
        begin=time.perf_counter()
        pixels=(decoder.decode_tiled(latent.float()) if req.tiled else decoder.decode(latent.float()))
        synchronize(device);decode_ms=(time.perf_counter()-begin)*1000
        _sessions[sid]=(identity,req.chunk,latent.detach().cpu().float())
        _sessions.move_to_end(sid)
        while len(_sessions)>16:_sessions.popitem(last=False)
        truth=torch.from_numpy(np.stack([np.array(Image.open(directory/"raw"/"chase"/f"{start+i:05d}.jpg").convert("RGB")) for i in range(8)])).permute(3,0,1,2)[None].to(device).float()/127.5-1
        from evaluate import pixel_metrics
        measured=pixel_metrics(pixels,truth)
        waypoints=points[0].float().cpu().tolist()
        return {"frames":video_urls(pixels),"cameras":camera_urls,"waypoints":waypoints,
                "projected_waypoints":project_waypoints(waypoints),"goal_xy":goal,"source":"Mean-Flow",
                "trained":True,"checkpoint_step":config["step"],"frame_start":start,"fps":36,
                "telemetry":{"dit_ms":dit_ms,"decode_ms":decode_ms,"policy_ms":policy_ms,
                             "effective_fps":8000/(dit_ms+decode_ms+policy_ms),"psnr_db":measured["psnr_db"],
                             "sram_mib":decoder.sram_budget_report()["total_mib"],
                             "sram_kind":"analytical INT8 weights + FP16 tile estimate",
                             "uve_stage_frames":[s["frame_id"] for s in engine.last_telemetry["stages"]],
                             "freshness_gain_ms":engine.last_telemetry["freshness_gain_ms"],
                             "stream_mode":"cached replay; staged frame arrivals emulated"}}


def main():
    import uvicorn
    uvicorn.run("app.server:app",host="127.0.0.1",port=8088,workers=1)


if __name__=="__main__":main()
