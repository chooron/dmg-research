#!/usr/bin/env python3
"""Wait for remote alpha=0.005 LORO completion, download final artifacts, shut down."""
from __future__ import annotations

import os
from pathlib import Path
import posixpath
import re
import stat
import time

import paramiko


HOST = "connect.westb.seetacloud.com"
PORT = 53700
USER = "root"
REMOTE_ROOT = "/root/dmg-research/project/flexmopex/results/block3_loro/config_dmopex_v1"
LOCAL_ROOT = Path("/home/jingxin/code/dmg-research/project/flexmopex/results/block3_loro/config_dmopex_v1")
LOG = Path("/home/jingxin/code/dmg-research/project/flexmopex/logs/block3_remote_monitor.log")
EXPECTED = [(4, 456), (5, 42), (5, 123), (5, 456), (6, 42), (6, 123), (6, 456)]


def log(message: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def connect() -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        HOST,
        port=PORT,
        username=USER,
        password=os.environ["REMOTE_PASS"],
        timeout=20,
        auth_timeout=20,
        banner_timeout=20,
    )
    return client


def status(client: paramiko.SSHClient) -> tuple[int, bool]:
    command = (
        "ROOT=/root/dmg-research/project/flexmopex/results/block3_loro/config_dmopex_v1; "
        "find \"$ROOT\" -path '*flex_alpha_0_005_region[456]/seed_*/sim/metrics_agg.json' "
        "-type f | wc -l; "
        "ps -eo cmd= | grep -E 'run_block3_loro_resume|run_model.py.*alpha 0.005' "
        "| grep -v grep | wc -l"
    )
    _, stdout, _ = client.exec_command(command, timeout=20)
    values = stdout.read().decode(errors="replace").split()
    return int(values[0]), int(values[1]) > 0


def wanted(relative: str) -> bool:
    if relative.endswith("/model/learnedweightmopex_ep50.pt"):
        return True
    return bool(
        re.search(
            r"/test[^/]*_Ep50/(w_(int|phen|snow|sub)\.npy|metrics\.json|metrics_agg\.json)$",
            relative,
        )
    )


def download(client: paramiko.SSHClient) -> tuple[int, int]:
    sftp = client.open_sftp()
    selected: list[tuple[str, str, int]] = []

    def walk(remote_dir: str, relative: str) -> None:
        for attr in sftp.listdir_attr(remote_dir):
            remote_path = posixpath.join(remote_dir, attr.filename)
            rel = posixpath.join(relative, attr.filename)
            if stat.S_ISDIR(attr.st_mode):
                walk(remote_path, rel)
            elif stat.S_ISREG(attr.st_mode) and wanted(rel):
                selected.append((remote_path, rel, attr.st_size))

    for region, seed in EXPECTED:
        run_rel = f"flex_alpha_0_005_region{region}/seed_{seed}"
        remote_run = posixpath.join(REMOTE_ROOT, run_rel)
        try:
            walk(remote_run, run_rel)
        except OSError as exc:
            log(f"download listing failed for {run_rel}: {exc}")
            raise

    total = sum(size for _, _, size in selected)
    log(f"downloading {len(selected)} files ({total} bytes)")
    for index, (remote_path, relative, _) in enumerate(selected, 1):
        local_path = LOCAL_ROOT / relative
        local_path.parent.mkdir(parents=True, exist_ok=True)
        sftp.get(remote_path, str(local_path))
        if index % 12 == 0 or index == len(selected):
            log(f"downloaded {index}/{len(selected)}")
    sftp.close()
    return len(selected), total


def shut_down() -> None:
    client = connect()
    try:
        log("all remote runs complete; issuing shutdown")
        client.exec_command("shutdown -h now", timeout=10)
    finally:
        client.close()


def main() -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    log("monitor started")
    while True:
        try:
            client = connect()
            try:
                count, active = status(client)
            finally:
                client.close()
            log(f"remote complete={count}/{len(EXPECTED)}, active={active}")
            if count == len(EXPECTED) and not active:
                client = connect()
                try:
                    download(client)
                finally:
                    client.close()
                shut_down()
                log("shutdown command sent")
                return
        except Exception as exc:  # noqa: BLE001
            log(f"poll/download error: {type(exc).__name__}: {exc}")
        time.sleep(60)


if __name__ == "__main__":
    main()
