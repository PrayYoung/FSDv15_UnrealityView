"""Select the actual validation-best evaluation for post-run baseline reproduction."""
import argparse
import json
from pathlib import Path
from unrealityview.data.cache import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--train',required=True);p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.train)
    step=json.loads((root/'checkpoint_best/config.json').read_text())['step']
    report=json.loads((root/f'eval_{step:06d}/metrics.json').read_text())
    atomic_json(a.output,report)


if __name__=='__main__':main()
