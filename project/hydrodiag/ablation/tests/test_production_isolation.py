import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = PROJECT_ROOT / "ablation/runners/run_xnes_ablation.py"
CONFIG_PATH = PROJECT_ROOT / "ablation/configs/ic_xnes_stage1_screening_v1.json"


def test_no_import_of_production_runner() -> None:
    code = RUNNER_PATH.read_text()
    import_lines = [
        line
        for line in code.splitlines()
        if line.startswith("import ") or line.startswith("from ")
    ]
    for line in import_lines:
        assert "run_xnes_production" not in line


def test_no_read_of_559() -> None:
    code = RUNNER_PATH.read_text()
    assert "559sub_id.txt" not in code


def test_no_call_of_old_training_ic_data() -> None:
    code = RUNNER_PATH.read_text()
    assert "training.ic.data" not in code
    assert "training/ic/data.py" not in code


def test_no_write_to_old_results() -> None:
    code = RUNNER_PATH.read_text()
    assert "results/ic_xnes_full" not in code
    assert "stage1_preflight" not in code


def test_config_uses_a_split_and_no_test_metric() -> None:
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    assert cfg["split"] == "A"
    assert cfg["n_basins"] == 32
    assert cfg["compute_test_metric"] is False
