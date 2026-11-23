from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def to_image(frame):
    if hasattr(frame,"detach"):
        frame = frame.detach().float().cpu().clamp(-1,1).permute(1,2,0).numpy()
        frame = np.rint((frame+1)*127.5).astype(np.uint8)
    return Image.fromarray(frame).convert("RGB")


def contact_sheet(columns, path, frame_indices=(0,3,7)):
    """columns: [(label, C×T×H×W)], with honest GT/prediction labelling."""
    w,h = columns[0][1].shape[-1],columns[0][1].shape[-2]
    out = Image.new("RGB",(len(columns)*w,len(frame_indices)*(h+28)),"#111821")
    draw = ImageDraw.Draw(out)
    for col,(label,video) in enumerate(columns):
        for row,frame in enumerate(frame_indices):
            y = row*(h+28)
            draw.text((col*w+8,y+7),f"{label} / f{frame}",fill="white")
            out.paste(to_image(video[:,frame]),(col*w,y+28))
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    out.save(path)


def save_rollout(video, path, fps=36):
    frames = [to_image(video[:,i]) for i in range(video.shape[1])]
    frames[0].save(path,save_all=True,append_images=frames[1:],duration=1000/fps,loop=0)
