"""
Remote Execution & Monitoring Script.
Uploads latest scripts and launches 6 parallel dPL models on remote GPU node (connect.westb.seetacloud.com:53700).
Monitors initial progress and reports status.
"""
import os
import time
from pathlib import Path
import paramiko

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
LOCAL_REPO = BENCHMARK_ROOT.parents[1]

REMOTE_REPO = "/root/dmg-research"
REMOTE_BENCHMARK = f"{REMOTE_REPO}/project/benchmark"

MODELS_6 = ["simhyd", "hbv96", "gr4j", "collie1", "wetland", "alpine1"]


def sftp_mkdir_p(sftp, remote_directory):
    dirs_to_create = []
    current_dir = remote_directory
    while current_dir and current_dir != "/":
        dirs_to_create.append(current_dir)
        current_dir = os.path.dirname(current_dir)
    dirs_to_create.reverse()
    for dir_path in dirs_to_create:
        try:
            sftp.stat(dir_path)
        except IOError:
            sftp.mkdir(dir_path)


def main():
    print("=== Connecting to Remote SSH Server (connect.westb.seetacloud.com:53700) ===")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        hostname="connect.westb.seetacloud.com",
        port=53700,
        username="root",
        password="SWsjA1FU+8H1",
        timeout=30,
    )
    sftp = ssh.open_sftp()

    # 1. Upload latest scripts to remote
    scripts_to_upload = [
        "run_dpl_benchmark_20ep.py",
        "run_dpl_benchmark_dmg_native.py",
        "launch_parallel_dpl_remote.py",
        "launch_36models_pool6.py",
    ]
    for script_name in scripts_to_upload:
        loc_path = BENCHMARK_ROOT / "scripts" / script_name
        rem_path = f"{REMOTE_BENCHMARK}/scripts/{script_name}"
        sftp_mkdir_p(sftp, os.path.dirname(rem_path))
        print(f"Syncing script -> {rem_path}")
        sftp.put(str(loc_path), rem_path)

    sftp.close()

    # Kill old test processes if any to ensure clean pool startup
    print("\nCleaning up any previous test processes...")
    ssh.exec_command("pkill -9 -f run_dpl_benchmark_dmg_native; pkill -9 -f run_dpl_benchmark; pkill -9 -f launch_36models_pool6; pkill -9 -f python3")
    time.sleep(3)

    # 2. Launch 36-Model Master Pool Manager (Constant 6 Workers) in Nohup Background
    nohup_cmd = (
        f"export PATH=/root/miniconda3/bin:$PATH && "
        f"cd {REMOTE_BENCHMARK} && "
        f"PYTHONPATH='{REMOTE_BENCHMARK}:{REMOTE_BENCHMARK}/src:$PYTHONPATH' "
        f"nohup python3 scripts/launch_36models_pool6.py --epochs 20 --device cuda --max_workers 6 > logs/master_launch.nohup 2>&1 &"
    )

    print(f"\nExecuting Nohup Master Pool Command:\n{nohup_cmd}\n")
    ssh.exec_command(nohup_cmd)

    # 3. Monitor Initial Progress & Check GPU Utilization
    print("Waiting 6 seconds for Master Pool to fill 6 worker processes...")
    time.sleep(6)

    # Check remote processes & nvidia-smi
    stdin, stdout, stderr = ssh.exec_command("export PATH=/root/miniconda3/bin:$PATH; nvidia-smi && ps aux | grep run_dpl_benchmark | grep -v grep")
    print("\nRemote GPU Status & Active 6 Worker Processes:\n", stdout.read().decode("utf-8"))

    # Read Master Pool Log
    stdin, stdout, stderr = ssh.exec_command(f"cat {REMOTE_BENCHMARK}/logs/dpl_pool/master_pool.log 2>/dev/null || cat {REMOTE_BENCHMARK}/logs/master_launch.nohup")
    print("\nMaster Pool Log Output:\n", stdout.read().decode("utf-8").strip())

    ssh.close()


if __name__ == "__main__":
    main()
