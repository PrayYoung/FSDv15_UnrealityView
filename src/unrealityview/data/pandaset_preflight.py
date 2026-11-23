"""Validate camera metadata before renting compute; never repair or reorder clocks."""
import numpy as np
from .real_world import PANDA_CAMERAS,check_timestamps,pose_matrix


def validate_camera_metadata(cameras,frames=80,max_skew=.06):
    times=[]
    for name in PANDA_CAMERAS:
        camera=cameras[name]
        try:
            intr=camera['intrinsics'];values=np.array([intr[k] for k in ('fx','fy','cx','cy')],float)
            if not np.isfinite(values).all() or min(values[:2])<=0:raise ValueError('Invalid intrinsics')
            if len(camera['timestamps'])<frames or len(camera['poses'])<frames:raise ValueError('Incomplete metadata')
            if camera['image_indices']!=list(range(frames)):raise ValueError('Incomplete/duplicate image indices')
            t=check_timestamps(camera['timestamps'][:frames])
            for pose in camera['poses'][:frames]:
                if not np.isfinite(pose_matrix(pose)).all():raise ValueError('Nonfinite pose')
            times.append(t)
        except (ValueError,KeyError,TypeError) as error:raise ValueError(f'{name}: {error}') from error
    front=times[1]
    for name,t in zip(PANDA_CAMERAS,times):
        if np.abs(front[:,None]-t[None,:]).min(1).max()>max_skew:
            raise ValueError(f'{name}: camera skew exceeds bound')
    return dict(frames=frames,fps=float(1/np.median(np.diff(front))))
