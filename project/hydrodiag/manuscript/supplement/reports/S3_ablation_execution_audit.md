# S3 Ablation Execution Audit

The remote host is actively running the IC XNES full-model series calibration launcher. It is not running a dPL process in this audit snapshot. The declared remote task family is incomplete: 8,234/47,790 result artifacts were present. Seven remote dPL-like directories were found under the separate data outputs root and matched to local smoke/legacy directories by config hash; no formal 531 multiseed dPL production result was verified.

Remote host: autodl-container-491e4b9a34-2d10558e
Active command: /root/miniconda3/bin/python ablation/controlled_optimizer_ablation/run_remote_full_model_series.py 
Output: /root/outputs/full_model_series_calibration/v1/tasks
Progress: 8,234/47,790; failure markers: 0.
Local dPL directory count: 7.
Local classification counts: {'SMOKE': 3, 'LEGACY_559': 4}.

S3.1 has partial, qualified optimizer/task evidence. S3.2 calibration sweep evidence is unresolved. S3.3 has partial artifact coverage and does not support an unqualified adequacy claim. S3.4 is supported by local production code. S3.5 formal multiseed evidence is missing. S3.6 remains subject to the remote period conflict. S3.7 has a host/GPU snapshot but not a complete reproducibility manifest.

The selected 30-second remote stat sample was unchanged; this was limited to the sampled files and was not interpreted as a stopped run. The active process table still contained the master plus three GPU workers.

This audit was read-only on the remote host. No running process was modified and no remote file was changed.
