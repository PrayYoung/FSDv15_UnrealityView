import pytest
import torch
from unrealityview.explicit3d.project import composite_live_first,project
from unrealityview.explicit3d.preprocess import preprocess_rgb,depth_input_shape
from tools.chase_3dgs.explicit_two_timestamp import hole_stats


def test_live_priority_union_and_invalid_pixels_are_exact():
    current=torch.tensor([[[1.,0,0],[1,0,0]],[[1,0,0],[1,0,0]]])
    past=torch.tensor([[[0.,1,0],[0,1,0]],[[0,1,0],[0,1,0]]])
    live=torch.tensor([[True,False],[False,True]]);historical=torch.tensor([[True,True],[False,False]])
    output,mask=composite_live_first(current,live,past,historical)
    assert torch.equal(output[live],current[live])
    assert torch.equal(mask,torch.tensor([[True,True],[False,True]]))
    assert torch.equal(output[0,1],past[0,1]) and torch.equal(output[1,0],torch.zeros(3))
    assert torch.equal(current,torch.tensor([1.,0,0]).expand_as(current))


def test_historical_world_pose_fills_only_the_current_hole():
    # Old camera was one metre to the right in world space; project into current
    # target frame, not into the old ego frame or image-aligned pixel coordinates.
    k=torch.eye(3)[None];current_pose=torch.eye(4)[None];past_pose=current_pose.clone();past_pose[0,0,3]=1
    depth=torch.tensor([[[1.,float('nan')]]]);current_rgb=torch.tensor([[[[1.,0,0],[0,0,0]]]])
    past_rgb=torch.tensor([[[[0.,1,0],[0,0,0]]]])
    live,m,_=project(current_rgb,depth,k,current_pose,current_pose[0],k[0],(2,1))
    past,mp,_=project(past_rgb,depth,k,past_pose,current_pose[0],k[0],(2,1))
    output,mask=composite_live_first(live,m,past,mp)
    assert mask.all();torch.testing.assert_close(output,torch.tensor([[[1.,0,0],[0,1,0]]]))


def test_priority_does_not_claim_to_reject_stale_content_in_current_holes():
    # Deliberately stale historical car color still fills a current hole. This
    # guards the claim boundary: covered-current identity is not ghost rejection.
    now=torch.zeros(1,1,3);past=torch.ones_like(now)
    rgb,m=composite_live_first(now,torch.zeros(1,1,dtype=torch.bool),past,torch.ones(1,1,dtype=torch.bool))
    assert m.all() and torch.equal(rgb,past)
    with pytest.raises(ValueError,match='boolean'):
        composite_live_first(now,torch.zeros(1,1),past,torch.ones(1,1))


def test_batched_rgb_preprocessing_shape_normalization_and_no_camera_mixing():
    assert depth_input_shape(384,640,518)==(518,868)
    assert depth_input_shape(384,640,392)==(392,658)
    source=torch.zeros(2,12,20,3);source[1]=1
    result=preprocess_rgb(source,28)
    assert result.shape==(2,3,28,42)
    mean=torch.tensor([.485,.456,.406]);std=torch.tensor([.229,.224,.225])
    torch.testing.assert_close(result[0,:,10,10],-mean/std)
    torch.testing.assert_close(result[1,:,10,10],(1-mean)/std)
    with pytest.raises(ValueError):preprocess_rgb(torch.zeros(2,3,12,20),28)


def test_hole_metrics_use_the_scaled_original_road_band():
    mask=torch.ones(384,640,dtype=torch.bool);mask[134:]=False
    stats=hole_stats(mask)
    assert stats['lower_band']==1 and stats['whole']==pytest.approx(250/384)
    improved=mask.clone();improved[134:250]=True
    assert hole_stats(improved)['whole']<stats['whole']
