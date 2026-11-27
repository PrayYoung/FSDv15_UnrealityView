"""Shared exact frame/pose definitions for the two authorized scene040 probes."""
import json,math,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from prepare_chase_teacher import chase_calibration
from unrealityview.data.real_world import CV_TO_FLU
from unrealityview.data.cache import atomic_json

FRAMES=list(range(12,80))
SAMPLES=[12,25,40,55,70]
SIZE=(640,384)
POSES={'A':dict(back=6.8,height_m=3.8,pitch=19.),'B':dict(back=5.5,height_m=3.,pitch=17.)}


def chase_pose(front,name,size=SIZE,yaw=0):
    pose,k=chase_calibration(front,width=size[0],height=size[1],**POSES[name])
    if yaw:
        pivot=front.copy();pivot[:3,:3]=front[:3,:3]@CV_TO_FLU.T
        a=math.radians(yaw);rotation=np.eye(4);rotation[:2,:2]=[[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]]
        pose=pivot@rotation@np.linalg.inv(pivot)@pose
    return pose,k


def preview(image,size):
    # Letterbox when aspect ratios differ; never silently stretch calibration.
    return ImageOps.pad(image,size,method=Image.Resampling.LANCZOS,color='black')


def comparison(columns,labels,frames=SAMPLES):
    sheet=Image.new('RGB',(320*len(columns),220*len(frames)),'#111820');draw=ImageDraw.Draw(sheet)
    for row,index in enumerate(frames):
        for col,(source,label) in enumerate(zip(columns,labels)):
            sheet.paste(preview(source[index],(320,192)),(col*320,row*220+28))
            draw.text((col*320+4,row*220+5),f'{label} / {index}',fill='white')
    return sheet


def delta(images):
    arrays=[np.asarray(images[i].resize((320,192))).astype(np.float32)/255 for i in FRAMES]
    return float(np.abs(np.diff(np.stack(arrays),axis=0)).mean())


def seal_views(root,groups,references,timestamps,extra=None):
    root=Path(root)
    record=dict(frames=FRAMES,size=list(SIZE),groups=groups,references=references,
        timestamps=[float(timestamps[i]) for i in FRAMES],extra=extra or {})
    atomic_json(root/'views.json',record)


def load_pil(path):
    with Image.open(path) as image:return image.convert('RGB')
