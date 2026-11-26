import torch
from torch import nn
import torch.nn.functional as F


def pseudo_huber(prediction,target,c=.1):
    # Recipe-v2 normalization (the v1 expression was smaller by a factor c).
    return (torch.sqrt((prediction.float()-target.float()).square()+c*c)-c).mean()


def charbonnier(prediction,target,eps=1e-3):
    return torch.sqrt((prediction.float()-target.float()).square()+eps*eps).mean()


class ChannelNormalizedVGG16(nn.Module):
    def __init__(self,features=None):
        super().__init__()
        if features is None:
            from torchvision.models import vgg16,VGG16_Weights
            features=vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:23]
        self.features=features.eval().requires_grad_(False)
        self.register_buffer('mean',torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer('std',torch.tensor([.229,.224,.225])[None,:,None,None])

    def forward(self,prediction,target):
        x=(prediction.clamp(0,1)-self.mean)/self.std
        y=(target.clamp(0,1)-self.mean)/self.std;loss=x.new_zeros(())
        weights={3:1.,8:1.,15:1.,22:.5}
        for index,layer in enumerate(self.features):
            x=layer(x)
            with torch.no_grad():y=layer(y)
            if index in weights:
                xn=F.normalize(x.float(),dim=1,eps=1e-6)
                yn=F.normalize(y.float(),dim=1,eps=1e-6)
                loss=loss+weights[index]*(xn-yn).square().sum(dim=1).mean()
        return loss/3.5


def three_term_loss(pred_z,target_z,pred_rgb,target_rgb,perceptual):
    terms={'pseudo_huber':pseudo_huber(pred_z,target_z),'charbonnier':2*charbonnier(pred_rgb,target_rgb),
           'vgg':perceptual(pred_rgb,target_rgb)}
    return .5*terms['pseudo_huber']+terms['charbonnier']+.8*terms['vgg'],terms


def sobel_l1(pred,target):
    k=pred.new_tensor([[-1,0,1],[-2,0,2],[-1,0,1]])/8
    gray=pred.mean(1,keepdim=True);truth=target.mean(1,keepdim=True)
    return sum(F.l1_loss(F.conv2d(gray,axis[None,None],padding=1),F.conv2d(truth,axis[None,None],padding=1)) for axis in (k,k.T))/2
