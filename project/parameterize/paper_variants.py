from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from project.parameterize.implements import build_paper_dpl as _build_paper_dpl

_PAPER_VARIANT_TO_NN = {
    "deterministic": "DeterministicParamModel",
    "mc_dropout": "McMlpModel",
    "distributional": "DistributionalParamModel",
}


def _normalize_seed_list(raw_value: Any) -> list[int]:
    if raw_value is None:
        return []
    if isinstance(raw_value, int):
        return [int(raw_value)]
    if isinstance(raw_value, str):
        tokens = raw_value.replace(",", " ").split()
        return [int(token) for token in tokens]
    return [int(value) for value in raw_value]


def normalize_paper_config(raw_config: dict[str, Any]) -> None:
    """Normalize paper-facing config into the paper-stack runtime contract."""
    paper_cfg = raw_config.setdefault("paper", {})
    variant = str(paper_cfg.get("variant", "mc_dropout")).lower()
    if variant not in _PAPER_VARIANT_TO_NN:
        raise ValueError(
            f"Unsupported paper.variant '{variant}'. Expected one of: "
            f"{', '.join(sorted(_PAPER_VARIANT_TO_NN))}."
        )

    split = str(paper_cfg.get("split", "main")).lower()
    if split != "main":
        raise ValueError(
            f"Unsupported paper.split '{split}'. Slice 1 only supports 'main'."
        )

    seeds = _normalize_seed_list(paper_cfg.get("seeds", raw_config.get("seeds")))
    if not seeds:
        seeds = [int(raw_config.get("seed", 42))]
    paper_cfg["variant"] = variant
    paper_cfg["split"] = split
    paper_cfg["seeds"] = seeds

    raw_config["seed"] = int(raw_config.get("seed", seeds[0]))
    raw_config["seeds"] = seeds
    raw_config["trainer"] = "MyTrainer"
    raw_config.setdefault("data_loader", "HydroLoader")
    raw_config.setdefault("data_sampler", "HydroSampler")

    model_cfg = raw_config.setdefault("model", {})
    phy_cfg = model_cfg.setdefault("phy", {})
    nn_cfg = model_cfg.setdefault("nn", {})
    nn_cfg["name"] = _PAPER_VARIANT_TO_NN[variant]
    nn_cfg.setdefault("forcings", [])
    nn_cfg.setdefault("attributes", [])
    nn_cfg["output_activation"] = "sigmoid"
    nn_cfg.setdefault("static_pool", "last")
    phy_cfg["nmul"] = int(phy_cfg.get("nmul", 1) or 1)

    distribution_cfg = raw_config.setdefault("distribution", {})
    distribution_cfg.setdefault("beta_kl", 1e-3)
    distribution_cfg.setdefault("kl_warmup_epochs", 10)
    nn_distribution_cfg = nn_cfg.setdefault("distribution", {})
    nn_distribution_cfg.setdefault("logstd_min", -5.0)
    nn_distribution_cfg.setdefault("logstd_max", 2.0)

    test_cfg = raw_config.setdefault("test", {})
    if variant == "mc_dropout":
        test_cfg["mc_dropout"] = True
        test_cfg["mc_samples"] = max(int(test_cfg.get("mc_samples", 100)), 1)
    else:
        test_cfg["mc_dropout"] = False
        test_cfg["mc_samples"] = 1


def validate_paper_config(config: dict[str, Any]) -> None:
    """Validate the normalized paper config for the parameterize paper stack."""
    paper_cfg = config.get("paper", {})
    variant = str(paper_cfg.get("variant", "")).lower()
    if variant not in _PAPER_VARIANT_TO_NN:
        raise ValueError(f"Unknown paper.variant '{variant}'.")

    if str(config.get("trainer", "")) != "MyTrainer":
        raise ValueError("Paper stack must target MyTrainer.")

    data_cfg = config.get("data", {})
    for key in ("basin_ids_path", "basin_ids_reference_path"):
        if not data_cfg.get(key):
            raise ValueError(
                f"Paper stack requires data.{key} for MyTrainer runtime."
            )

    model_cfg = config.get("model")
    if not isinstance(model_cfg, dict):
        raise ValueError("Paper stack requires a 'model' mapping in the config.")

    nn_cfg = model_cfg.get("nn")
    if not isinstance(nn_cfg, dict):
        raise ValueError("Paper stack requires a 'model.nn' mapping in the config.")

    phy_cfg = model_cfg.get("phy")
    if not isinstance(phy_cfg, dict):
        raise ValueError("Paper stack requires a 'model.phy' mapping in the config.")

    if nn_cfg.get("forcings"):
        raise ValueError(
            "Paper stack expects static basin attributes only; set model.nn.forcings to []."
        )
    if not nn_cfg.get("attributes"):
        raise ValueError(
            "Paper stack requires at least one static attribute in model.nn.attributes."
        )
    if str(nn_cfg.get("output_activation", "sigmoid")).lower() != "sigmoid":
        raise ValueError(
            "Paper stack expects output_activation='sigmoid' for the HBV parameter boundary."
        )
    if int(phy_cfg.get("nmul", 1) or 1) != 1:
        raise ValueError("Paper stack currently requires model.phy.nmul == 1.")


def build_paper_dpl(config: dict[str, Any]):
    """Thin wrapper around the parameterize-local DPL builder."""
    return _build_paper_dpl(config)


def write_run_metadata(config: dict[str, Any]) -> str:
    """Write a flat JSON summary for one paper-stack run."""
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    paper_cfg = config.get("paper", {})
    loss_cfg = config.get("train", {}).get("loss_function", {})
    test_cfg = config.get("test", {})
    nn_cfg = config.get("model", {}).get("nn", {})
    data_cfg = config.get("data", {})
    payload = {
        "paper_variant": str(paper_cfg.get("variant", "")),
        "seed": int(config["seed"]),
        "split": str(paper_cfg.get("split", "main")),
        "nn_name": str(nn_cfg.get("name", "")),
        "loss_name": str(loss_cfg.get("name", "unknown")),
        "mc_samples": int(test_cfg.get("mc_samples", 1)),
        "output_activation": str(nn_cfg.get("output_activation", "")),
        "static_pool": str(nn_cfg.get("static_pool", "")),
        "paper_seeds": list(_normalize_seed_list(paper_cfg.get("seeds", []))),
        "data_basin_ids_path": str(data_cfg.get("basin_ids_path", "")),
    }
    path = output_dir / "run_meta.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return os.fspath(path)
