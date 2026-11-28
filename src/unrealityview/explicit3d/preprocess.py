"""Batched device-resident RGB resize/normalization for pinned metric DAv2."""
import torch
import torch.nn.functional as F


def depth_input_shape(height,width,input_size):
    if min(height,width,input_size)<=0:raise ValueError('Positive image dimensions required')
    scale=max(input_size/height,input_size/width)
    def multiple(value):
        result=round(value/14)*14
        if result<input_size:result=((input_size+13)//14)*14
        return int(result)
    return multiple(height*scale),multiple(width*scale)


def preprocess_rgb(images,input_size):
    """C,H,W,3 RGB in[0,1] -> C,3,H14,W14 on the original device.

    Matches the upstream lower-bound/multiple-of14 size policy; PyTorch bicubic
    replaces OpenCV's CPU resize. No per-camera CPU loop or BGR conversion.
    """
    if images.ndim!=4 or images.shape[-1]!=3 or not images.is_floating_point():
        raise ValueError('Expected floating RGB C,H,W,3 in[0,1]')
    h,w=images.shape[1:3];shape=depth_input_shape(h,w,input_size)
    resized=F.interpolate(images.permute(0,3,1,2),size=shape,mode='bicubic',align_corners=False)
    mean=images.new_tensor([.485,.456,.406]).reshape(1,3,1,1)
    std=images.new_tensor([.229,.224,.225]).reshape(1,3,1,1)
    return (resized-mean)/std
