"""Modern SegFormer masks in DriveStudio's exact NNN_camera.png layout."""
import argparse
import json
from pathlib import Path
from PIL import Image
import numpy as np


def main():
    import torch
    from transformers import AutoImageProcessor,SegformerForSemanticSegmentation
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args()
    spec=json.loads((Path(__file__).parent/'upstream_pins.json').read_text())['sky_model']
    processor=AutoImageProcessor.from_pretrained(spec['repo'],revision=spec['revision'])
    model=SegformerForSemanticSegmentation.from_pretrained(spec['repo'],revision=spec['revision'],use_safetensors=False,weights_only=True).eval().cuda()
    if str(model.config.id2label.get(10,'')).lower()!='sky':raise ValueError('Model label 10 is not sky')
    scene=Path(a.scene);output=scene/'sky_masks';output.mkdir(exist_ok=True)
    paths=sorted((scene/'images').glob('*.jpg'))
    if not paths:raise ValueError('No processed camera images')
    with torch.no_grad():
        for path in paths:
            with Image.open(path) as image:image=image.convert('RGB')
            inputs=processor(images=image,return_tensors='pt').to('cuda')
            logits=model(**inputs).logits
            labels=torch.nn.functional.interpolate(logits,size=(image.height,image.width),mode='bilinear',align_corners=False).argmax(1)[0]
            Image.fromarray((labels.eq(10).cpu().numpy().astype(np.uint8)*255)).save(output/(path.stem+'.png'))
    (output/'source.json').write_text(json.dumps(spec,indent=2)+'\n')


if __name__=='__main__':main()
