#!/usr/bin/env python3
"""Build an exact, data-free OOB deployment snapshot and checksums."""
from __future__ import annotations

import argparse
import csv
import os
import tarfile
from pathlib import Path

from oob_common import (
    BENCHMARK_ROOT,
    PRIMARY8,
    git_sha,
    load_oob_config,
    resolve_repo_path,
    sha256_file,
    write_json,
)


def required_files(config: dict, assignment_path: Path) -> list[Path]:
    paths = [
        BENCHMARK_ROOT / "scripts/oob/__init__.py",
        BENCHMARK_ROOT / "scripts/oob/oob_common.py",
        BENCHMARK_ROOT / "scripts/oob/make_oob_folds.py",
        BENCHMARK_ROOT / "scripts/oob/run_oob_job.py",
        BENCHMARK_ROOT / "scripts/oob/run_oob_queue.py",
        BENCHMARK_ROOT / "scripts/oob/oob_status.py",
        BENCHMARK_ROOT / "scripts/oob/summarize_oob.py",
        BENCHMARK_ROOT / "scripts/oob/prepare_oob_deployment.py",
        BENCHMARK_ROOT / "scripts/oob/prepare_oob_data.py",
        BENCHMARK_ROOT / "scripts/oob/oob_preflight.py",
        Path(config["_resolved_from"]),
        BENCHMARK_ROOT / "dpl/attributes.py",
        BENCHMARK_ROOT / "dpl/nn_parameterizer.py",
        BENCHMARK_ROOT / "dpl/optimizer_transaction.py",
        BENCHMARK_ROOT / "src/checkpointing.py",
        BENCHMARK_ROOT / "src/data_selection.py",
        BENCHMARK_ROOT / "src/model_registry.py",
        BENCHMARK_ROOT / "src/objective.py",
        BENCHMARK_ROOT.parents[1] / "project/benchmark/tests/test_oob_spatial_generalization.py",
    ]
    paths.extend(sorted((BENCHMARK_ROOT.parents[1] / "dmotpy").rglob("*.py")))
    paths.append(assignment_path)
    return paths


def archive_path(path: Path) -> Path:
    relative = path.relative_to(BENCHMARK_ROOT.parents[1])
    return relative


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="oob_primary8_5fold_20260902.yaml")
    parser.add_argument("--out", default=None, help="Deployment artifact directory")
    args = parser.parse_args()
    config = load_oob_config(args.config)
    config_path = Path(config["_resolved_from"]).resolve()
    try:
        config_path.relative_to(BENCHMARK_ROOT.parents[1])
    except ValueError as exc:
        raise SystemExit("deployment config must be inside the repository so the snapshot is self-contained") from exc
    out = resolve_repo_path(args.out or "project/benchmark/results/oob_deploy_20260902")
    out.mkdir(parents=True, exist_ok=True)
    assignment = resolve_repo_path(config["folds"]["assignment_file"])
    files = required_files(config, assignment)
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise SystemExit("deployment inputs missing:\n" + "\n".join(missing))

    model_list = out / "OOB_PRIMARY8_MODELS.txt"
    model_list.write_text("\n".join(PRIMARY8) + "\n", encoding="utf-8")
    files.append(model_list)
    manifest_path = out / "OOB_DEPLOY_MANIFEST.csv"
    rows = []
    for path in files:
        rows.append({"path": str(archive_path(path)), "size": path.stat().st_size, "sha256": sha256_file(path)})
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "size", "sha256"])
        writer.writeheader()
        writer.writerows(rows)

    tar_path = out / "oob_deploy_20260902.tar.gz"
    with tarfile.open(tar_path, "w:gz") as archive:
        for path in files:
            arcname = archive_path(path)
            info = archive.gettarinfo(str(path), arcname=str(arcname))
            info.mtime = 0
            if path.is_file():
                with path.open("rb") as handle:
                    archive.addfile(info, handle)
            else:
                archive.addfile(info)
    checksum_path = out / "OOB_DEPLOY_SHA256.txt"
    checksum_path.write_text(f"{sha256_file(tar_path)}  {tar_path.name}\n", encoding="utf-8")
    write_json(
        out / "OOB_DEPLOY_METADATA.json",
        {
            "experiment_id": config["experiment_id"],
            "git_sha": git_sha(),
            "manifest": str(manifest_path),
            "archive": str(tar_path),
            "archive_sha256": sha256_file(tar_path),
            "file_count": len(files),
            "data_included": False,
            "git_included": False,
            "models": list(PRIMARY8),
        },
    )
    print(f"manifest={manifest_path}")
    print(f"archive={tar_path}")
    print(f"sha256={sha256_file(tar_path)}")


if __name__ == "__main__":
    main()
