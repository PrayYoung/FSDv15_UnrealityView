"""Coverage and appearance policies, independent of external reconstruction code."""
from contextlib import contextmanager
import numpy as np

GATE_POLICY='dynamic-teacher-ground-band-v2'


def alpha_coverages(alpha):
    alpha=np.asarray(alpha)
    if alpha.shape!=(192,320) or not np.isfinite(alpha).all() or (alpha<0).any() or (alpha>1).any():
        raise ValueError('Expected finite 192x320 opacity in [0,1]')
    occupied=alpha>=.5
    return dict(whole=float(occupied.mean()),ground=float(occupied[67:,:].mean()),
                ego=float(occupied[144:,112:208].mean()))


@contextmanager
def consistent_novel_appearance(trainer,frames):
    """Force mean held-out appearance only during chase rendering, then restore."""
    original=trainer.test_set_indices
    flags=[(model,model.in_test_set) for model in trainer.models.values() if hasattr(model,'in_test_set')]
    try:
        trainer.test_set_indices=set(range(frames))
        yield
    finally:
        trainer.test_set_indices=original
        for model,value in flags:model.in_test_set=value
