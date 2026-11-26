"""Small opt-in receipt after a completed upstream optimizer update; stdlib only."""
import json
import math
import os
from pathlib import Path
from datetime import datetime,timezone


def record_optimizer_progress(step,losses):
    target=os.environ.get('UNREALITY_PROGRESS_FILE')
    if not target or (step not in (0,1,9) and step%100):return
    values={k:float(v.detach().item()) if hasattr(v,'detach') else float(v) for k,v in losses.items()}
    if not all(math.isfinite(v) for v in values.values()):raise ValueError('Nonfinite optimizer receipt loss')
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(phase='optimizing',step_index=int(step),completed_updates=int(step)+1,losses=values,
        observed_at=datetime.now(timezone.utc).isoformat(),note='Written after trainer.backward(), which includes optimizer_step in pinned DriveStudio')
    temporary=target.with_suffix('.tmp');temporary.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n');temporary.replace(target)
