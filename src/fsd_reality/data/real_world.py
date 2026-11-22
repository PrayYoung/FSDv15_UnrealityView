"""Camera-only PandaSet and Waymo Perception v2 readers. No TensorFlow or pickle.

Canonical camera slots: front_left, front, front_right, left, rear, right.
Camera geometry is OpenCV (right, down, forward); ego is (forward,left,up).
"""
from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
import numpy as np
from PIL import Image

CAMERA_SLOTS=("front_left","front","front_right","left","rear","right")
PANDA_CAMERAS=("front_left_camera","front_camera","front_right_camera","left_camera","back_camera","right_camera")
WAYMO_CAMERAS=(2,1,3,4,None,5)
# OpenCV camera axes -> forward/left/up. Waymo extrinsics use FLU camera axes.
CV_TO_FLU=np.array([[0.,0,1],[-1,0,0],[0,-1,0]])


@dataclass
class CameraFrame:
    timestamp: float
    images: list
    intrinsics: np.ndarray
    camera_to_ego: np.ndarray
    camera_mask: np.ndarray
    ego_pose: np.ndarray
    camera_timestamps: np.ndarray


def pose_matrix(pose):
    h=pose["heading"]
    w,x,y,z=[float(h[k]) for k in ("w","x","y","z")]
    norm=np.linalg.norm([w,x,y,z])
    if not np.isfinite(norm) or norm<1e-8:raise ValueError("Invalid pose quaternion")
    w,x,y,z=np.array([w,x,y,z])/norm
    rotation=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                       [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                       [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    result=np.eye(4);result[:3,:3]=rotation
    result[:3,3]=[pose["position"][k] for k in ("x","y","z")]
    return result


def resize_calibrated(image,k,size=(224,140),distortion=None):
    """Rectify before resizing. Returned K refers to the resized RGB image."""
    if distortion is not None and np.any(np.asarray(distortion)!=0):
        import cv2
        image=Image.fromarray(cv2.undistort(np.asarray(image),k,np.asarray(distortion)))
    width,height=image.size
    resized=k.copy().astype(np.float64)
    resized[0]*=size[0]/width;resized[1]*=size[1]/height
    return image.resize(size,Image.Resampling.BILINEAR),resized


def check_timestamps(times,max_gap=.25):
    times=np.asarray(times,dtype=np.float64)
    if len(times)<2 or not np.isfinite(times).all() or (np.diff(times)<=0).any():
        raise ValueError("Timestamps must be finite and strictly increasing")
    if np.diff(times).max()>max_gap:raise ValueError("Sequence has a camera gap; do not splice it")
    return times


class PandaSetSequence:
    def __init__(self,root,sequence,max_skew=.06):
        self.directory=Path(root)/sequence
        self.max_skew=max_skew
        self.cameras=[]
        for name in PANDA_CAMERAS:
            directory=self.directory/"camera"/name
            intr=json.loads((directory/"intrinsics.json").read_text())
            k=np.array([[intr["fx"],0,intr["cx"]],[0,intr["fy"],intr["cy"]],[0,0,1]],float)
            poses=json.loads((directory/"poses.json").read_text())
            times=check_timestamps(json.loads((directory/"timestamps.json").read_text()))
            paths=sorted(directory.glob("*.jpg"))
            if len(paths)!=len(times) or len(poses)!=len(times):raise ValueError(f"Incomplete PandaSet camera {name}")
            self.cameras.append((k,poses,times,paths))
        self.timestamps=self.cameras[1][2]
        # PandaSet publishes sensor poses, not an exact vehicle/IMU reference.
        # Use the FRONT camera center as reference and its orientation converted
        # to FLU; relative waypoint XY is still physically well defined.
        self.reference_kind="front_camera_center_FLU"

    def __len__(self):return len(self.timestamps)

    def frame(self,index):
        timestamp=self.timestamps[index]
        front=pose_matrix(self.cameras[1][1][index])
        world_from_ego=front.copy();world_from_ego[:3,:3]=front[:3,:3]@CV_TO_FLU.T
        images,ks,extrinsics,times=[],[],[],[]
        for k,poses,ts,paths in self.cameras:
            nearest=int(np.abs(ts-timestamp).argmin())
            if abs(ts[nearest]-timestamp)>self.max_skew:raise ValueError("PandaSet camera skew exceeds bound")
            with Image.open(paths[nearest]) as img:image=img.convert("RGB")
            image,k=resize_calibrated(image,k)
            images.append(image);ks.append(k);times.append(ts[nearest])
            extrinsics.append(np.linalg.inv(world_from_ego)@pose_matrix(poses[nearest]))
        yaw=np.arctan2(world_from_ego[1,0],world_from_ego[0,0])
        return CameraFrame(float(timestamp),images,np.stack(ks),np.stack(extrinsics),np.ones(6,bool),
                           np.array([*world_from_ego[:2,3],yaw]),np.asarray(times))


def parquet_rows(path,columns=None):
    import pyarrow.parquet as pq
    table=pq.ParquetFile(path)
    for batch in table.iter_batches(batch_size=64,columns=columns):yield from batch.to_pylist()


class WaymoSequence:
    """Read one native Perception v2 context without LiDAR, labels or TensorFlow."""
    def __init__(self,root,sequence,max_skew=.06):
        root=Path(root);self.max_skew=max_skew
        self.calibration={int(r["key.camera_name"]):r for r in parquet_rows(root/"camera_calibration"/(sequence+".parquet"))}
        self.poses={int(r["key.frame_timestamp_micros"]):np.asarray(r["[VehiclePoseComponent].world_from_vehicle.transform"]).reshape(4,4)
                    for r in parquet_rows(root/"vehicle_pose"/(sequence+".parquet"))}
        # Index row groups; keep only one decoded row group in memory.
        import pyarrow.parquet as pq
        self.parquet=pq.ParquetFile(root/"camera_image"/(sequence+".parquet"))
        self.groups={};self._loaded_group=-1;self._loaded_rows=[]
        for group in range(self.parquet.num_row_groups):
            rows=self.parquet.read_row_group(group,columns=["key.frame_timestamp_micros","key.camera_name"]).to_pylist()
            for row_index,row in enumerate(rows):
                stamp=int(row["key.frame_timestamp_micros"]);camera=int(row["key.camera_name"])
                if camera in self.groups.setdefault(stamp,{}):raise ValueError("Duplicate Waymo camera key")
                self.groups[stamp][camera]=(group,row_index)
        self.keys=sorted(self.groups)
        if not self.keys or any(set(self.groups[t])!=set(range(1,6)) or t not in self.poses for t in self.keys):
            raise ValueError("Waymo context lacks synchronized five-camera data or vehicle poses")
        self.timestamps=check_timestamps(np.asarray(self.keys)/1e6)
        self.reference_kind="Waymo_vehicle_FLU"

    def __len__(self):return len(self.keys)

    def _row(self,location):
        group,row=location
        if group!=self._loaded_group:
            self._loaded_rows=self.parquet.read_row_group(group).to_pylist();self._loaded_group=group
        return self._loaded_rows[row]

    def frame(self,index):
        stamp=self.keys[index];vehicle=self.poses[stamp]
        images,ks,extrinsics,times=[],[],[],[]
        for name in WAYMO_CAMERAS:
            if name is None:
                images.append(Image.new("RGB",(224,140)));ks.append(np.eye(3));extrinsics.append(np.eye(4));times.append(stamp/1e6)
                continue
            row=self._row(self.groups[stamp][name]);cal=self.calibration[name]
            prefix="[CameraCalibrationComponent].intrinsic."
            k=np.array([[cal[prefix+"f_u"],0,cal[prefix+"c_u"]],[0,cal[prefix+"f_v"],cal[prefix+"c_v"]],[0,0,1]],float)
            distortion=[cal[prefix+x] for x in ("k1","k2","p1","p2","k3")]
            with Image.open(BytesIO(row["[CameraImageComponent].image"])) as im:image=im.convert("RGB")
            image,k=resize_calibrated(image,k,distortion=distortion)
            camera_pose=np.asarray(row["[CameraImageComponent].pose.transform"],float).reshape(4,4)
            native_ext=np.asarray(cal["[CameraCalibrationComponent].extrinsic.transform"],float).reshape(4,4)
            cv_transform=np.eye(4);cv_transform[:3,:3]=CV_TO_FLU
            ext=np.linalg.inv(vehicle)@camera_pose@native_ext@cv_transform
            ts=float(row["[CameraImageComponent].pose_timestamp"])
            if abs(ts-stamp/1e6)>self.max_skew:raise ValueError("Waymo camera pose skew exceeds bound")
            images.append(image);ks.append(k);extrinsics.append(ext);times.append(ts)
        yaw=np.arctan2(vehicle[1,0],vehicle[0,0])
        return CameraFrame(stamp/1e6,images,np.stack(ks),np.stack(extrinsics),np.array([1,1,1,1,0,1],bool),
                           np.array([*vehicle[:2,3],yaw]),np.asarray(times))
