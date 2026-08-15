"""R3 controlled synthetic-truth experiment (XAJ-CN generating structure).

R3 answers, on a known-truth synthetic experiment over the full CAMELS-531
basin set:

1. whether Base/TGD2 recover the synthetic discharge after re-calibration,
2. whether shared XAJ parameters deviate additionally from the generating
   truth when discharge is recovered,
3. whether common internal states deviate from the generating truth,
4. how these compensation traces differ between IC-CMA-ES and dPL.

The module is intentionally thin: it reuses the repository loaders
(``ablation.ic_core``), model implementations (``models``), the IC-CMA-ES
pipeline (``training.ic``) and the dPL pipeline (``training.dpl``).
"""

from . import common  # noqa: F401
