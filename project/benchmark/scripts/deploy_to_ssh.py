"""
Deployment & Sync Script for Remote SSH Server (connect.westb.seetacloud.com:53700).
Syncs benchmark code, dPL module, scripts, configs, and 35-dim Caravan attributes to remote /root/dmg-research.
"""
import os
import sys
from pathlib import Path
import paramiko

# Local paths
LOCAL_REPO = Path("/home/jingxin/code/dmg-research")
LOCAL_BENCHMARK = LOCAL_REPO / "project" / "benchmark"
LOCAL_DATA = LOCAL_REPO / "data"

# Remote paths
REMOTE_REPO = "/root/dmg-research"
REMOTE_BENCHMARK = f"{REMOTE_REPO}/project/benchmark"
REMOTE_DATA = f"{REMOTE_REPO}/data"

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

def upload_file(sftp, local_path, remote_path):
    remote_dir = os.path.dirname(remote_path)
    sftp_mkdir_p(sftp, remote_dir)
    print(f"Uploading {local_path.name} -> {remote_path}")
    sftp.put(str(local_path), remote_path)

def main():
    print("=== Connecting to Remote SSH Node (connect.westb.seetacloud.com:53700) ===")
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

    # 1. Sync Data files
    data_files = [
        "caravan_671_attributes.npy",
        "caravan_671_attributes.csv",
        "531sub_id.txt",
        "gage_id.npy",
        "camels_dataset",
    ]
    for df in data_files:
        loc_f = LOCAL_DATA / df
        if loc_f.exists():
            upload_file(sftp, loc_f, f"{REMOTE_DATA}/{df}")
        else:
            print(f"Warning: {loc_f} not found locally.")

    # 2. Sync Benchmark Code & dPL Files
    sync_dirs = ["dpl", "src", "dmotpy", "scripts", "configs"]
    for sdir in sync_dirs:
        loc_dir = LOCAL_BENCHMARK / sdir
        if loc_dir.exists():
            for root, dirs, files in os.walk(loc_dir):
                for f in files:
                    if f.endswith((".py", ".json", ".yaml", ".sh", ".txt", ".csv")):
                        l_file = Path(root) / f
                        rel_p = l_file.relative_to(LOCAL_BENCHMARK)
                        r_file = f"{REMOTE_BENCHMARK}/{rel_p}"
                        upload_file(sftp, l_file, r_file)

    print("\n=== All Files Successfully Uploaded & Synced ===")

    # Verify remote CUDA environment
    stdin, stdout, stderr = ssh.exec_command(
        "python3 -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}\")'"
    )
    print("Remote PyTorch Verification:", stdout.read().decode("utf-8").strip())

    sftp.close()
    ssh.close()

if __name__ == "__main__":
    main()
