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
from pydantic import BaseModel,ConfigDict,Field,field_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware
from unrealityview.checkpoint import load_checkpoint
from unrealityview.data.cache import resolve_safe_subpath,load_episode,episode_grid
from unrealityview.data.dataset import stage_uve_tokens
from unrealityview.device import select_device,synchronize,autocast_for
from unrealityview.geometry import ground_raycast,project_waypoints,ego_waypoints
from unrealityview.models.pretrained_taesd import PretrainedTAESD
from unrealityview.runtime.ring_buffer_engine import UVERingBuffer,RealityViewStreamingEngine
from unrealityview.models.lora import LoRAWeightManager,load_lora

APP_DIR=Path(__file__).resolve().parent
DATA_ROOT=Path(os.environ.get("UNREALITYVIEW_DATA","outputs/real_dataset")).resolve()
CHECKPOINT=Path(os.environ.get("UNREALITYVIEW_CHECKPOINT","outputs/real_train/checkpoint")).resolve()
LORA_DIR=Path(os.environ.get("UNREALITYVIEW_LORA_DIR","outputs/adapters")).resolve()
STYLES_DATA_DIR=Path(os.environ.get("UNREALITYVIEW_STYLES_DATA","outputs/styles")).resolve()
PORT=int(os.environ.get("UNREALITYVIEW_PORT","8088"))
app=FastAPI(title="UnrealityView",docs_url=None,redoc_url=None)
app.add_middleware(TrustedHostMiddleware,allowed_hosts=["127.0.0.1","localhost","testserver"])
app.mount("/static",StaticFiles(directory=APP_DIR/"static"),name="static")
_lock=threading.Lock()
_models=None
_episode_cache=OrderedDict()
_sessions=OrderedDict()
_style_manager=None
_adapter_cache={}


class TapGoal(BaseModel):
    model_config=ConfigDict(extra="forbid",allow_inf_nan=False,strict=True)
    u:float=Field(ge=0,le=320)
    v:float=Field(ge=0,le=192)


class RenderRequest(BaseModel):
    model_config=ConfigDict(extra="forbid",allow_inf_nan=False,strict=True)
    episode:str=Field(min_length=1,max_length=120)
    chunk:int=Field(default=1,ge=1,le=10000)
    nfe:Literal[1,2,4]=1
    source:Literal["model","ground_truth"]="ground_truth"
    rollout:Literal["teacher_forced","closed_loop"]="teacher_forced"
    interleaving:bool=True
    tiled:bool=False
    style:Literal["reality","comic","cyberpunk","hybrid"]="reality"
    style_strength:float=Field(default=1.,ge=0,le=1.5)
    closed_loop:bool=False
    goal:TapGoal|None=None
    session_id:str=Field(pattern=r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
    seed:int=Field(default=42,ge=0,le=2**31-1)

    @field_validator("nfe",mode="before")
    @classmethod
    def integer_nfe(cls,value):
        if type(value) is not int:
            raise ValueError("nfe must be an integer, not a boolean or float")
        return value


@app.middleware("http")
async def local_security(request:Request,call_next):
    origin=request.headers.get("origin")
    if origin and origin not in (f"http://127.0.0.1:{PORT}",f"http://localhost:{PORT}"):
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
            "styles_available":available_styles(),"purpose":"Independent research prototype; no vehicle control"}


def adapter_path(style):
    for relative in (f"{style}_lora.safetensors",f"{style}/{style}_lora.safetensors"):
        path=resolve_safe_subpath(LORA_DIR,relative)
        if path.is_file():return path
    return None


def available_styles():
    styles=["reality"]+[style for style in ("comic","cyberpunk") if adapter_path(style) is not None]
    if "comic" in styles and "cyberpunk" in styles:styles.append("hybrid")
    return styles


def apply_style(model,style,strength):
    global _style_manager,_adapter_cache
    if _style_manager is None or _style_manager.model is not model:
        _style_manager=LoRAWeightManager(model);_adapter_cache={}
    requested=[]
    names=[] if style=="reality" or strength==0 else ["comic","cyberpunk"] if style=="hybrid" else [style]
    for name in names:
        path=adapter_path(name)
        if path is None:raise HTTPException(409,f"No trained {name} adapter is installed")
        key=(str(path),path.stat().st_mtime_ns,path.stat().st_size)
        if key not in _adapter_cache:
            try:
                adapter=load_lora(path,expected_base=_style_manager.base_sha256)
                if adapter.style!=name:raise ValueError("Adapter style identity mismatch")
                _adapter_cache={k:v for k,v in _adapter_cache.items() if k[0]!=str(path)}
                _adapter_cache[key]=adapter
            except (ValueError,KeyError,OSError) as error:raise HTTPException(409,f"Incompatible {name} adapter: {error}")
        requested.append((name,_adapter_cache[key],strength*(.55 if style=="hybrid" else 1.)))
    try:metrics=_style_manager.apply_adapters(requested)
    except ValueError as error:raise HTTPException(422,str(error))
    return {"active_style":style if names else "reality","style_strength":strength,
            "lora_size_mb":metrics["adapter_mib"],"lora_merge_ms":metrics["merge_ms"],
            "lora_cache_hit":metrics["cache_hit"]}


@app.get("/api/episodes")
def episodes():
    rows=[]
    for path in sorted(DATA_ROOT.glob("*/*/episode.safetensors")):
        try:
            metadata=json.loads(path.with_name("metadata.json").read_text())
            rows.append({"id":str(path.parent.relative_to(DATA_ROOT)),"frames":metadata["frames"],
                         "source":metadata.get("source","unknown"),"fps":metadata.get("fps",36),
                         "target_view":metadata.get("target_view","elevated_chase")})
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


@app.get("/api/style-assets/{style}/{path:path}")
def style_asset(style:str,path:str):
    if style not in ("comic","cyberpunk"):raise HTTPException(404,"Style reference not found")
    try:target=resolve_safe_subpath(STYLES_DATA_DIR/style,path)
    except ValueError as error:raise HTTPException(400,str(error))
    if target.suffix!=".png" or not target.is_file():raise HTTPException(404,"Style reference image not found")
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
            raise HTTPException(409,"No trained checkpoint. Load a trained checkpoint or select reference frames.")
        device=select_device()
        model,policy,config=load_checkpoint(CHECKPOINT,device)
        decoder=PretrainedTAESD().to(device).eval()
        model.requires_grad_(False);policy.requires_grad_(False)
        _models=(model,policy,decoder,device,config)
    return _models


@app.post("/api/render")
@torch.no_grad()
def render(req:RenderRequest):
    if req.source!="model" and req.style!="reality" and (req.style=="hybrid" or req.style_strength!=1.):
        raise HTTPException(422,"Style references support comic/cyberpunk at strength 1; hybrid blending applies to model output")
    try: directory=resolve_safe_subpath(DATA_ROOT,req.episode)
    except ValueError as e: raise HTTPException(400,str(e))
    goal=None if req.goal is None else ground_raycast(req.goal.u,req.goal.v)
    if req.goal is not None and goal is None:
        raise HTTPException(422,"Tap a visible road point below the horizon (0–100m ahead)")
    start=req.chunk*8
    cameras=("front_left","front","front_right","rear_left","rear","rear_right")
    if not (directory/"episode.safetensors").is_file():
        raise HTTPException(404,"Episode cache missing; import camera data with prepare_real_data.py first")
    # Serialize model access and each user's closed-loop state. No browser-side
    # hidden state is trusted for model tensors or paths.
    with _lock:
        episode=get_episode(directory)
        if start+8>len(episode["chase_latents"]): raise HTTPException(422,"Chunk exceeds episode length")
        metadata_path=directory/"metadata.json"
        metadata=json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
        dataset_source=metadata.get("source","unknown cache")
        fps=float(metadata.get("fps",36))
        target_view=metadata.get("target_view","elevated_chase")
        camera_names=metadata.get("camera_names",("front_left","front","front_right","rear_left","rear","rear_right"))
        times=episode.get("timestamps",torch.arange(len(episode["uve_tokens"]),dtype=torch.float64)/fps)
        completed=episode["camera_timestamps"].max(1).values if "camera_timestamps" in episode else times
        available=episode.get("camera_mask",torch.ones(len(times),6,dtype=torch.bool))[start].tolist()
        if req.goal is not None and target_view!="elevated_chase":
            raise HTTPException(422,"Tap-to-drive raycasting requires an elevated chase target; unavailable in the native-front baseline")
        relative=str(directory.relative_to(DATA_ROOT))
        camera_urls=[f"/api/assets/{relative}/raw/{c}/{start:05d}.jpg" for c in cameras]
        camera_frames=[[f"/api/assets/{relative}/raw/{c}/{start+i:05d}.jpg" for c in cameras] for i in range(8)]
        if req.source=="ground_truth":
            waypoints=ego_waypoints(episode["ego_states"].float(),start).tolist()
            reference_frames=[f"/api/assets/{relative}/raw/chase/{start+i:05d}.jpg" for i in range(8)]
            if req.style!="reality":
                teacher=resolve_safe_subpath(STYLES_DATA_DIR/req.style,relative)
                if not (teacher/"frames"/f"{start:05d}.png").is_file():raise HTTPException(409,"Style reference cache is not installed")
                reference_frames=[f"/api/style-assets/{req.style}/{relative}/frames/{start+i:05d}.png" for i in range(8)]
            return {"frames":reference_frames,
                    "cameras":camera_urls,"camera_frames":camera_frames,"waypoints":waypoints,"projected_waypoints":project_waypoints(waypoints) if target_view=="elevated_chase" else [],
                    "goal_xy":goal,"source":"reference frames" if req.style=="reality" else req.style+" style reference","dataset_source":dataset_source,"trained":False,"frame_start":start,"fps":fps,
                    "target_view":target_view,"camera_names":camera_names,"camera_available":available,
                    "telemetry":{"dit_ms":None,"decode_ms":None,"policy_ms":None,"effective_fps":None,
                                 "active_style":req.style,"style_strength":req.style_strength,"lora_merge_ms":0.,"lora_size_mb":0.,
                                 "psnr_db":None,"sram_mib":None,"uve_stage_frames":[],"stream_mode":"cached ground truth"}}
        model,policy,decoder,device,config=get_models()
        grid=episode_grid(episode,metadata)
        if tuple(getattr(model,"uve_grid",(4,8)))!=grid:raise HTTPException(409,"Checkpoint/cache UVE grid mismatch")
        include_rays=model.config.geometry_conditioning
        style_metrics=apply_style(model,req.style,req.style_strength)
        previous=episode["chase_latents"][start-8:start].permute(1,0,2,3)[None].float().to(device)
        sid=str(req.session_id)
        identity=(req.episode,req.nfe,req.seed,req.interleaving,req.tiled)
        old=_sessions.get(sid)
        if (req.closed_loop or req.rollout=="closed_loop") and req.chunk>1:
            if old is None or old[0]!=identity or old[1] not in (req.chunk-1,req.chunk):
                raise HTTPException(409,"Closed-loop playback must begin at chunk 1 and advance sequentially. Reset playback.")
            previous=old[2 if old[1]==req.chunk-1 else 3].to(device)
        ring=UVERingBuffer()
        first=min(start-1,int(torch.searchsorted(completed,times[start],right=True))-1)
        ring.publish(stage_uve_tokens(episode,first,grid,include_rays)[None].to(device),ts=float(completed[first]),frame_id=first)
        engine=RealityViewStreamingEngine(model,ring)
        stage_current=[float(times[start])]
        newest_frame=[first]
        def stage_hook(stage):
            anchor=min(start+stage*2,len(times)-1)
            latest=int(torch.searchsorted(completed,times[anchor],right=True))-1
            frame=max(newest_frame[0],min(start+stage*2-1,latest))
            newest_frame[0]=frame
            stage_current[0]=max(stage_current[0],float(times[min(start+stage*2,len(times)-1)]))
            if req.interleaving:
                ring.publish(stage_uve_tokens(episode,frame,grid,include_rays)[None].to(device),ts=float(completed[frame]),frame_id=frame)
        generator=torch.Generator(device=device).manual_seed(req.seed+req.chunk)
        noise=torch.randn(previous.shape,device=device,generator=generator)
        synchronize(device);begin=time.perf_counter()
        with autocast_for(device):
            points=policy(episode["uve_tokens"][start][None].float().to(device),None if goal is None else torch.tensor([goal],device=device))
        synchronize(device);policy_ms=(time.perf_counter()-begin)*1000
        begin=time.perf_counter()
        with autocast_for(device):
            latent=engine.sample_chunk(previous,req.nfe,req.interleaving,noise=noise,
                         clock=lambda:stage_current[0],stage_hook=stage_hook)
        synchronize(device);dit_ms=(time.perf_counter()-begin)*1000
        begin=time.perf_counter()
        pixels=(decoder.decode_tiled(latent.float()) if req.tiled else decoder.decode(latent.float()))
        synchronize(device);decode_ms=(time.perf_counter()-begin)*1000
        _sessions[sid]=(identity,req.chunk,latent.detach().cpu().float(),previous.detach().cpu().float())
        _sessions.move_to_end(sid)
        while len(_sessions)>16:_sessions.popitem(last=False)
        reference="real_reference";paths=[directory/"raw/chase"/f"{start+i:05d}.jpg" for i in range(8)]
        active=style_metrics["active_style"]
        if active in ("comic","cyberpunk"):
            styled=resolve_safe_subpath(STYLES_DATA_DIR/active,relative)/"frames"
            paths=[styled/f"{start+i:05d}.png" for i in range(8)]
            reference="style_teacher" if all(path.is_file() for path in paths) else "unavailable"
        elif active=="hybrid":reference="unavailable"
        measured={"psnr_db":None}
        if reference!="unavailable":
            truth=torch.from_numpy(np.stack([np.array(Image.open(path).convert("RGB")) for path in paths])).permute(3,0,1,2)[None].to(device).float()/127.5-1
            from evaluate import pixel_metrics
            measured=pixel_metrics(pixels,truth)
        waypoints=points[0].float().cpu().tolist()
        return {"frames":video_urls(pixels),"cameras":camera_urls,"camera_frames":camera_frames,"waypoints":waypoints,
                "projected_waypoints":project_waypoints(waypoints) if target_view=="elevated_chase" else [],"goal_xy":goal,"source":"Mean-Flow",
                "trained":True,"checkpoint_step":config["step"],"dataset_source":dataset_source,
                "training_data_source":config.get("data_source","unknown"),"frame_start":start,"fps":fps,
                "target_view":target_view,"camera_names":camera_names,"camera_available":available,
                "telemetry":{"dit_ms":dit_ms,"decode_ms":decode_ms,"policy_ms":policy_ms,
                             "effective_fps":8000/max(dit_ms+decode_ms,1e-3),"psnr_db":measured["psnr_db"],
                             "psnr_reference":reference,"psnr_teacher_strength":1. if reference=="style_teacher" else None,
                             "request_compute_ms":dit_ms+decode_ms+policy_ms+style_metrics["lora_merge_ms"],
                             **style_metrics,
                             "sram_mib":decoder.sram_budget_report()["total_mib"],
                             "sram_kind":"analytical INT8 weights + FP16 tile estimate",
                             "uve_stage_frames":[s["frame_id"] for s in engine.last_telemetry["stages"]],
                             "freshness_gain_ms":engine.last_telemetry["freshness_gain_ms"],
                             "stream_mode":"cached replay; staged frame arrivals emulated"}}


def main():
    global LORA_DIR,STYLES_DATA_DIR,DATA_ROOT,CHECKPOINT,PORT
    import argparse
    import uvicorn
    parser=argparse.ArgumentParser(description="UnrealityView local cockpit")
    parser.add_argument("--lora-dir",default=str(LORA_DIR));parser.add_argument("--styles-data-dir",default=str(STYLES_DATA_DIR))
    parser.add_argument("--data",default=str(DATA_ROOT));parser.add_argument("--checkpoint",default=str(CHECKPOINT))
    parser.add_argument("--port",type=int,default=PORT)
    args=parser.parse_args()
    if not 1<=args.port<=65535:parser.error("port must be 1–65535")
    DATA_ROOT=Path(args.data).resolve();CHECKPOINT=Path(args.checkpoint).resolve();PORT=args.port
    os.environ["UNREALITYVIEW_DATA"]=str(DATA_ROOT);os.environ["UNREALITYVIEW_CHECKPOINT"]=str(CHECKPOINT)
    os.environ["UNREALITYVIEW_PORT"]=str(PORT)
    LORA_DIR=Path(args.lora_dir).resolve();STYLES_DATA_DIR=Path(args.styles_data_dir).resolve()
    os.environ["UNREALITYVIEW_LORA_DIR"]=str(LORA_DIR);os.environ["UNREALITYVIEW_STYLES_DATA"]=str(STYLES_DATA_DIR)
    uvicorn.run("app.server:app",host="127.0.0.1",port=PORT,workers=1)


if __name__=="__main__":main()
