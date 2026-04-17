from __future__ import annotations


def apply_runtime_overrides(raw_config, args) -> None:
    paper_cfg = raw_config.setdefault("paper", {})
    if args.variant:
        paper_cfg["variant"] = args.variant
    if args.split:
        paper_cfg["split"] = args.split
    if args.mode:
        raw_config["mode"] = args.mode
    if args.device:
        raw_config["device"] = args.device
    if args.gpu_id is not None:
        raw_config["gpu_id"] = args.gpu_id
    if args.seed is not None:
        raw_config["seed"] = args.seed
    if args.seeds is not None:
        paper_cfg["seeds"] = args.seeds
    if args.mc_samples is not None:
        raw_config.setdefault("test", {})
        raw_config["test"]["mc_samples"] = args.mc_samples
    if args.epochs is not None:
        original_epochs = raw_config["train"]["epochs"]
        raw_config["train"]["epochs"] = args.epochs
        lr_cfg = raw_config["train"].get("lr_scheduler")
        if (
            isinstance(lr_cfg, dict)
            and lr_cfg.get("name") == "CosineAnnealingLR"
            and ("T_max" not in lr_cfg or lr_cfg.get("T_max") == original_epochs)
        ):
            lr_cfg["T_max"] = args.epochs
