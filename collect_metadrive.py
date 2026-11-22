"""Capture all seven attached cameras from the same graphics frame, then cache.

uv run --extra simulator collect_metadrive.py --output outputs/dataset
"""
import argparse
import math
import platform
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from fsd_reality.data.cache import save_episode,atomic_json
from fsd_reality.device import select_device,autocast_for
from fsd_reality.models.mini_uve import MiniUVE
from fsd_reality.models.pretrained_taesd import PretrainedTAESD

CAMERAS = ("front_left","front","front_right","rear_left","rear","rear_right")
HEADINGS = (60,0,-60,120,180,-120)


def capture_rgb(sensor):
    # Request RGB explicitly; Panda's native getRamImage is platform-dependent BGR(A).
    image = sensor.buffer.getDisplayRegion(1).getScreenshot()
    return np.frombuffer(image.getRamImageAs("RGB"),np.uint8).reshape(image.getYSize(),image.getXSize(),3)[::-1].copy()


def camera_calibration(sensor, parent, width=224, height=140):
    fov = float(sensor.get_lens().getFov()[0])
    focal = width/(2*math.tan(math.radians(fov)/2))
    k = torch.tensor([[focal,0,width/2],[0,focal,height/2],[0,0,1]],dtype=torch.float32)
    matrix = sensor.cam.getMat(parent)
    panda = torch.tensor([[matrix.getCell(j,i) for j in range(4)] for i in range(4)])
    cv_to_panda = torch.tensor([[1.,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]])
    panda_to_ego = torch.tensor([[0.,1,0,0],[-1,0,0,0],[0,0,1,0],[0,0,0,1]])
    return k,panda_to_ego@panda@cv_to_panda


@torch.no_grad()
def collect(args):
    from panda3d.core import loadPrcFileData
    display = "pandagl" if platform.system()=="Darwin" else "p3headlessgl"
    loadPrcFileData("",f"load-display {display}\naudio-library-name null\nsync-video false")
    from metadrive.envs.metadrive_env import MetaDriveEnv
    from metadrive.component.sensors.rgb_camera import RGBCamera
    from metadrive.policy.idm_policy import IDMPolicy
    root = Path(args.output)
    root.mkdir(parents=True,exist_ok=True)
    device = select_device(args.device)
    uve,vae = MiniUVE().to(device).eval(),PretrainedTAESD().to(device).eval()
    sensors = {name:(RGBCamera,224,140) for name in CAMERAS}
    sensors["chase"] = (RGBCamera,320,192)
    config = dict(use_render=False,image_observation=True,image_on_cuda=False,
                  sensors=sensors,vehicle_config=dict(image_source="front"),
                  agent_policy=IDMPolicy,physics_world_step_size=1/72,decision_repeat=2,
                  traffic_density=.22,map=5,num_scenarios=args.train_episodes+args.val_episodes,
                  start_seed=args.seed,window_size=(320,192),camera_fov=75,
                  show_terrain=False,show_skybox=True,show_interface=False,log_level=50,
                  horizon=args.frames+20)
    env = MetaDriveEnv(config)
    manifest = {"source":"MetaDrive","fps":36,"camera_names":CAMERAS,"frames":args.frames,
                "train_episodes":args.train_episodes,"val_episodes":args.val_episodes,"seed":args.seed,
                "capture":"seven attached cameras, one renderFrame, explicit RGB readback"}
    atomic_json(root/"manifest.json",manifest)
    try:
        for episode in range(args.train_episodes+args.val_episodes):
            split = "train" if episode < args.train_episodes else "val"
            directory = root/split/f"episode_{episode:04d}"
            if (directory/"episode.safetensors").exists():
                print(f"CACHE_EXISTS {directory}",flush=True)
                continue
            env.reset(seed=args.seed+episode)
            for name,heading in zip(CAMERAS,HEADINGS):
                env.engine.get_sensor(name).track(env.agent.origin,(0,0,1.6),(heading,0,0))
            env.engine.get_sensor("chase").track(env.agent.origin,(0,-6.8,3.8),(0,-19,0))
            ks,extrinsics = zip(*(camera_calibration(env.engine.get_sensor(n),env.agent.origin) for n in CAMERAS))
            ks,extrinsics = torch.stack(ks),torch.stack(extrinsics)
            for n in (*CAMERAS,"chase"):
                (directory/"raw"/n).mkdir(parents=True,exist_ok=True)
            tokens,latents,poses = [],[],[]
            for frame in range(args.frames):
                _,_,terminated,truncated,_ = env.step([0,0])
                env.engine.graphicsEngine.renderFrame()
                rgb = [capture_rgb(env.engine.get_sensor(n)) for n in (*CAMERAS,"chase")]
                if frame == 0:
                    sanity = Image.new("RGB",(3*320,3*220),"#101820")
                    draw = ImageDraw.Draw(sanity)
                    for i,(name,arr) in enumerate(zip((*CAMERAS,"chase"),rgb)):
                        x,y = (i%3)*320,(i//3)*220
                        draw.text((x+8,y+6),name,fill="white")
                        sanity.paste(Image.fromarray(arr).resize((320,192)),(x,y+24))
                    sanity.save(directory/"raw_preview.png")
                for name,arr in zip((*CAMERAS,"chase"),rgb):
                    Image.fromarray(arr).save(directory/"raw"/name/f"{frame:05d}.jpg",quality=90)
                cameras = torch.from_numpy(np.stack(rgb[:6])).permute(0,3,1,2).float()[None].to(device)/255
                chase = torch.from_numpy(rgb[6]).permute(2,0,1).float()[None,:,None].to(device)/127.5-1
                with autocast_for(device):
                    tokens.append(uve(cameras,ks[None].to(device),extrinsics[None].to(device))[0].half().cpu())
                    latents.append(vae.encode(chase)[0,:,0].half().cpu())
                pos = env.agent.position
                poses.append([float(pos[0]),float(pos[1]),float(env.agent.heading_theta)])
                if (terminated or truncated) and frame < args.frames-1:
                    raise RuntimeError(f"Episode ended at frame {frame}; do not pad or splice episodes")
            save_episode(directory,{"uve_tokens":torch.stack(tokens),"chase_latents":torch.stack(latents),
                         "ego_states":torch.tensor(poses),"intrinsics":ks,"camera_to_ego":extrinsics},
                         {**manifest,"episode":episode,"split":split})
            print(f"CAPTURED {directory} frames={len(tokens)}",flush=True)
    finally:
        env.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",default="outputs/dataset")
    p.add_argument("--train-episodes",type=int,default=50)
    p.add_argument("--val-episodes",type=int,default=4)
    p.add_argument("--frames",type=int,default=180)
    p.add_argument("--seed",type=int,default=1000)
    p.add_argument("--device",default="auto")
    a=p.parse_args()
    if a.train_episodes<1 or a.val_episodes<1 or a.frames<64:
        p.error("Need positive train/val counts and >=64 frames")
    collect(a)


if __name__=="__main__":
    main()
