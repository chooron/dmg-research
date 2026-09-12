# S3 Remote Run Status

Host: autodl-container-491e4b9a34-2d10558e
Remote time: 2026-07-28T12:20:40+08:00
Uptime:  12:20:40 up 19 days, 22:01,  0 users,  load average: 5.59, 6.92, 6.83
Active project copy: /autodl-fs/data/dmg_hydro_structure_diagnosis
Output root: /root/outputs/full_model_series_calibration/v1/tasks
Connection error: none
GPU snapshot: 0, NVIDIA GeForce RTX 3080 Ti, 12288, 9458, 100, 595.71.05
Remote dPL inventory: 7 directories under /autodl-fs/data/dmg_hydro_structure_diagnosis/outputs; their configuration hashes are compared with local outputs in s3_remote_local_comparison.csv.

Observed command: /root/miniconda3/bin/python ablation/controlled_optimizer_ablation/run_remote_full_model_series.py 

The launcher declares 47,790 tasks (531 basins x 10 models x 3 seeds x 3 starts). The snapshot contains 8,234 result.json, 8,234 done.txt, 8,234 checkpoints, and 0 failure markers. This is 17.23% of the declared task count. The active log shows XNES population 108, seed 101, generation 250/300; 159 in-flight units are inferred as one 53-basin chunk times three starts and are not counted as completed.

This is an active, incomplete optimizer calibration/ablation series, not a completed foundation production result. The launcher date protocol conflicts with result/runtime metadata, and remote HBV D=12 conflicts with the local audited D=13.

A bounded 30-second stat sample found no size or mtime change for the selected master log and one completed-task trace. This is limited evidence about those files only; the master and three GPU workers remained running, so it is not treated as task termination.

No running process was stopped, paused, reprioritized, or otherwise modified. No remote file was created, modified, or deleted.
