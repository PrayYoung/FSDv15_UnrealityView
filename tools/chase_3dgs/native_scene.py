"""Native scene reader used only in isolated GPU tools, never core loaders."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from unrealityview.data.real_world import pose_matrix
from tools.chase_3dgs.extract_pandaset_scene import CAMERAS


class NativeScene:
    def __init__(self,root,use_lidar=True,sequence='040'):
        if len(sequence)!=3 or not sequence.isdigit():raise ValueError('Invalid sequence')
        self.root=Path(root)/sequence;self.metadata=[]
        for name in CAMERAS:
            p=self.root/'camera'/name;k=json.loads((p/'intrinsics.json').read_text())
            self.metadata.append((p,np.array([[k['fx'],0,k['cx']],[0,k['fy'],k['cy']],[0,0,1.]],dtype=np.float32),json.loads((p/'poses.json').read_text()),np.array(json.loads((p/'timestamps.json').read_text()))))
        self.timestamps=self.metadata[0][3]
        self.lidar_times=np.array(json.loads((self.root/'lidar/timestamps.json').read_text())) if use_lidar else None
        if len(self.timestamps)!=80 or not (np.diff(self.timestamps)>0).all():raise ValueError('Expected80 ordered native frames')

    def cameras(self,index,size=(640,384)):
        images,poses,offsets=self.cameras_multires(index,(size,))
        rgb,k=images[size]
        return rgb,k,poses,offsets

    def cameras_multires(self,index,sizes):
        images={size:[] for size in sizes};ks={size:[] for size in sizes};poses=[];offsets=[]
        stamp=self.timestamps[index]
        for path,k,rows,times in self.metadata:
            i=int(np.abs(times-stamp).argmin());offset=float(times[i]-stamp)
            if abs(offset)>.06:raise ValueError('Camera timestamp mismatch')
            with Image.open(path/f'{i:02d}.jpg') as raw:
                image=raw.convert('RGB');w,h=image.size
                for size in sizes:
                    new=k.copy();new[0]*=size[0]/w;new[1]*=size[1]/h
                    images[size].append(np.asarray(image.resize(size,Image.Resampling.LANCZOS)).copy());ks[size].append(new)
            poses.append(pose_matrix(rows[i]));offsets.append(offset)
        return {size:(np.stack(images[size]),np.stack(ks[size])) for size in sizes},np.stack(poses),offsets

    def front(self,index,size):
        with Image.open(self.root/'camera/front_camera'/f'{index:02d}.jpg') as image:
            return image.convert('RGB').resize(size,Image.Resampling.LANCZOS)

    def lidar_world(self,index):
        # Pickle tables are consumed only here, in an isolated pinned native-data
        # environment after the complete source archive SHA256 was verified.
        if self.lidar_times is None:raise RuntimeError('LiDAR is disabled for this camera-only reader')
        import pandas as pd
        i=int(np.abs(self.lidar_times-self.timestamps[index]).argmin())
        if abs(self.lidar_times[i]-self.timestamps[index])>.06:raise ValueError('LiDAR timestamp mismatch')
        table=pd.read_pickle(self.root/'lidar'/f'{i:02d}.pkl')
        points=table[['x','y','z']].to_numpy(dtype=np.float32)
        return points[np.isfinite(points).all(1)]
