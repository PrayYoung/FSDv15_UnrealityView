from types import SimpleNamespace
import json
import numpy as np
from PIL import Image
import pytest
import torch
from unrealityview.models.meanflow_dit import ConditionalMeanFlowDiT,DiTConfig
from scripts.diagnose_base_blur import select_indices, donor_indices,measurements,aggregate,diagnose_split


def test_scene_sampling_and_donor_separation():
    class Dataset(SimpleNamespace):
        def __len__(self):return len(self.index)
    dataset=Dataset(split='train',index=[(0,8),(0,10),(1,8),(1,10),(2,8)],metadata=[{'source_id':str(i)} for i in range(3)])
    assert select_indices(dataset,3)==[0,2,4]
    assert donor_indices(dataset,[0,2,4])==[2,4,0]
    dataset.split='val';assert select_indices(dataset,3)==[0,1,2]
    dataset.split='test'
    with pytest.raises(ValueError,match='Reserved'):select_indices(dataset,3)


def test_per_frame_aggregation():
    target=torch.zeros(1,4,8,2,2);pred=target.clone();pred[:,:,7]=1
    truth=torch.zeros(1,3,8,2,2)
    m=measurements(pred,target,pred[:,:3],truth)
    assert m['latent_l1']==.125 and m['per_frame'][7]['latent_l1']==1
    assert m['per_frame'][0]['latent_l1']==0
    assert aggregate([m,m])['per_frame']==m['per_frame']


def test_inference_controls_are_finite_and_keep_camera_tokens(tmp_path):
    class Dataset:
        split='val';index=[(0,8),(1,8)]
        metadata=[{'source_id':'pandaset/001'},{'source_id':'pandaset/002'}]
        episodes=[{'timestamps':torch.arange(80)/10} for _ in range(2)]
        def __len__(self):return 2
        def __getitem__(self,i):
            return dict(z_prev=torch.ones(4,8,2,2)*(i+1)/10,z0=torch.zeros(4,8,2,2),
                uve_tokens=torch.ones(4,6,4)*(i+1),stage_ages=torch.ones(4)/10,
                episode=i,start=8,source_id=self.metadata[i]['source_id'],stage_frames=torch.tensor([7,9,11,13]))
    dataset=Dataset();dataset.paths=[]
    for i in range(2):
        root=tmp_path/str(i);raw=root/'raw/chase';raw.mkdir(parents=True)
        dataset.paths.append(root/'episode.safetensors')
        for j in range(8,16):Image.fromarray(np.full((2,2,3),128,dtype=np.uint8)).save(raw/f'{j:05d}.jpg')
    class Model:
        inner=ConditionalMeanFlowDiT(DiTConfig(hidden_dim=16,depth=4,num_heads=4,context_dim=4,latent_shape=(8,2,2)))
        def eval(self):self.inner.eval();return self
        def sample_chunk(self,previous,tokens,n,ages,noise):
            assert tokens.ne(0).any(-1).all() and not torch.is_grad_enabled()
            return self.inner.sample_chunk(previous,tokens,n,ages,noise=noise)
    class Decoder:
        def decode(self,z):return z[:,:3]
    report=diagnose_split(Model(),Decoder(),dataset,torch.device('cpu'),tmp_path,count=2)
    assert len(report['conditioning_1nfe'])==5
    assert len(report['nfe']['1']['per_frame'])==8
    assert (tmp_path/'val_contact_sheet.png').exists()
    json.dumps(report,allow_nan=False)
