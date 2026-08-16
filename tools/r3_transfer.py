#!/usr/bin/env python3
"""Robust file transfer from AutoDL remote via chunked base64 over ssh.

Verifies each file with sha256 after transfer. Retries chunks on failure.
Usage: transfer.py <manifest.json>   # manifest: list of {remote, local}
"""
import base64
import hashlib
import json
import os
import subprocess
import sys
import time

HOST = "connect.westb.seetacloud.com"
PORT = "42368"
SSH_BASE = [
    "setsid", "ssh", "-p", PORT,
    "-o", "StrictHostKeyChecking=no",
    "-o", "ConnectTimeout=15",
    "-o", "ServerAliveInterval=10",
    "-o", "ServerAliveCountMax=4",
    "-o", "KexAlgorithms=curve25519-sha256",
    "-o", "PreferredAuthentications=password",
    "-o", "PubkeyAuthentication=no",
    "-o", "NumberOfPasswordPrompts=1",
    f"root@{HOST}",
]
CHUNK_MB = 4  # binary MB per chunk -> ~5.3 MB base64 over the wire
MAX_ATTEMPTS = 6


def ssh_run(command: str, timeout: int = 300) -> bytes:
    env = dict(os.environ)
    env["SSH_ASKPASS"] = "/tmp/askpass.sh"
    env["SSH_ASKPASS_REQUIRE"] = "force"
    for attempt in range(MAX_ATTEMPTS):
        try:
            res = subprocess.run(
                SSH_BASE + [command],
                capture_output=True, timeout=timeout, env=env,
                stdin=subprocess.DEVNULL,
            )
            if res.returncode == 0:
                return res.stdout
            if attempt == MAX_ATTEMPTS - 1:
                raise RuntimeError(f"ssh failed rc={res.returncode}: {res.stderr[-300:]}")
        except subprocess.TimeoutExpired:
            if attempt == MAX_ATTEMPTS - 1:
                raise RuntimeError(f"ssh timeout: {command[:80]}")
        time.sleep(2 + attempt)
    raise RuntimeError("unreachable")


def remote_sha256(remote: str) -> str:
    out = ssh_run(f"sha256sum {remote} | cut -d' ' -f1").decode().strip()
    return out


def remote_size(remote: str) -> int:
    out = ssh_run(f"stat -c %s {remote}").decode().strip()
    return int(out)


def transfer_file(remote: str, local: str) -> dict:
    size = remote_size(remote)
    os.makedirs(os.path.dirname(local), exist_ok=True)
    total_chunks = (size + CHUNK_MB * 1024 * 1024 - 1) // (CHUNK_MB * 1024 * 1024)
    h = hashlib.sha256()
    start = time.time()
    with open(local, "wb") as f:
        for i in range(total_chunks):
            skip = i * CHUNK_MB
            cmd = (
                f"dd if={remote} bs=1048576 skip={skip} count={CHUNK_MB} 2>/dev/null | base64 -w0"
            )
            data = b""
            for attempt in range(MAX_ATTEMPTS):
                try:
                    out = ssh_run(cmd)
                    if out.strip():  # non-empty
                        data = base64.b64decode(out.strip())
                        break
                except Exception:
                    pass
                time.sleep(2)
            if not data:
                raise RuntimeError(f"chunk {i}/{total_chunks} failed for {remote}")
            f.write(data)
            h.update(data)
            if i % 8 == 0 or i == total_chunks - 1:
                elapsed = time.time() - start
                mb_done = (i + 1) * CHUNK_MB
                print(
                    f"  [{os.path.basename(remote)}] chunk {i+1}/{total_chunks} "
                    f"({mb_done}MB, {elapsed:.0f}s)", flush=True)
    local_sha = h.hexdigest()
    remote_sha = remote_sha256(remote)
    ok = local_sha == remote_sha
    return {
        "remote": remote, "local": local, "size": size,
        "local_sha256": local_sha, "remote_sha256": remote_sha,
        "match": ok, "elapsed_s": round(time.time() - start, 1),
    }


def main():
    manifest = json.load(open(sys.argv[1]))
    results = []
    for item in manifest:
        print(f"transfer: {item['remote']}", flush=True)
        results.append(transfer_file(item["remote"], item["local"]))
    all_ok = all(r["match"] for r in results)
    print(json.dumps(results, indent=1))
    print(f"ALL_MATCH={all_ok}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
