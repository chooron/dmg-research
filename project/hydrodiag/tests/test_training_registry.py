from training.dpl.run_dpl_model import (
    LITE_MODEL_REGISTRY,
)
from training.dpl.run_dpl_model import (
    MODEL_REGISTRY as DPL_REGISTRY,
)

EXPECTED = {
    "HBV",
    "GR4J",
    "XAJ",
    "SIMHYD",
    "GR4J_CN",
    "XAJ_CN",
    "SIMHYD_CN",
    "GR4J_PD",
    "XAJ_PD",
    "SIMHYD_PD",
    "XAJ_TGD2",
    "GR4J_TGD2",
    "SIMHYD_TGD2",
    "XAJ_2S",
    "XAJ_RWPE",
    "XAJ_D_E_CN",
    "XAJ_G_E_CN",
    "XAJ_D_R_CN",
    "XAJ_G_R_CN",
}


def test_active_dpl_registry_covers_all_models():
    assert set(DPL_REGISTRY) == EXPECTED
    assert set(LITE_MODEL_REGISTRY) == EXPECTED
