# R1 Inference Audit

{
  "batch_size": 64,
  "basin_count": 531,
  "basin_list": "/home/jingxin/code/dmg-research/data/531sub_id.txt",
  "conversion": "repository ablation.ic_core.data_adapter.convert_ft3s_to_mm_day; area_gages2 index 11",
  "dpl_mapping": "training.dpl.run_dpl_model.physical_parameters: sigmoid output; TGD2 residence times inverse-log mapped",
  "dpl_model_path": "training.dpl.run_dpl_model.LITE_MODEL_REGISTRY and StaticParameterNet",
  "device": "cuda",
  "execution": "five basin partitions were launched sequentially; each partition completed before the next and used GPU batched inference",
  "partition_count": 5,
  "dpl_normalization": "training.dpl.run_dpl_model.robust_normalize: selected-basin median/IQR, finite-value median fill, clip [-5,5]",
  "files": [
    {
      "basins": 531,
      "checkpoint_epoch": 100,
      "date_end": "2010-09-30",
      "date_start": "1980-10-01",
      "file": "/home/jingxin/code/dmg-research/project/hydro_structure_diagnosis/manuscript/results/R1/r1_daily_simulations_dpl_xaj_tgd2_seed_42.parquet",
      "model": "XAJ-TGD",
      "paradigm": "dPL-MLP",
      "periods": [
        "train",
        "test"
      ],
      "rows": 5624352,
      "seed_or_restart": "seed_42",
      "sha256": "7ceee80b99688deda7466f1174da9610c74a7f74d69eac68db33c4e82dc408ac",
      "source_selection": "latest common valid periodic checkpoint epoch 100",
      "status": "complete"
    },
    {
      "basins": 531,
      "checkpoint_epoch": 100,
      "date_end": "2010-09-30",
      "date_start": "1980-10-01",
      "file": "/home/jingxin/code/dmg-research/project/hydro_structure_diagnosis/manuscript/results/R1/r1_daily_simulations_dpl_xaj_tgd2_seed_123.parquet",
      "model": "XAJ-TGD",
      "paradigm": "dPL-MLP",
      "periods": [
        "train",
        "test"
      ],
      "rows": 5624352,
      "seed_or_restart": "seed_123",
      "sha256": "b68ef034b07ca0e5b6d39bf58c84bd70bf9e06df8a0062fad3c269ef95665b94",
      "source_selection": "latest common valid periodic checkpoint epoch 100",
      "status": "complete"
    },
    {
      "basins": 531,
      "checkpoint_epoch": 100,
      "date_end": "2010-09-30",
      "date_start": "1980-10-01",
      "file": "/home/jingxin/code/dmg-research/project/hydro_structure_diagnosis/manuscript/results/R1/r1_daily_simulations_dpl_xaj_tgd2_seed_2026.parquet",
      "model": "XAJ-TGD",
      "paradigm": "dPL-MLP",
      "periods": [
        "train",
        "test"
      ],
      "rows": 5624352,
      "seed_or_restart": "seed_2026",
      "sha256": "2d04bee5ebb6c4780554c74676721dba0d02da3d9db976300903d840a47dfe76",
      "source_selection": "latest common valid periodic checkpoint epoch 100",
      "status": "complete"
    }
  ],
  "forcing_order": [
    "P",
    "T",
    "PET"
  ],
  "forcing_source": "/home/jingxin/code/dmg-research/data/camels_dataset",
  "ic_model_path": "ablation.ic_core.model_adapter.ModelAdapter with variant=lite",
  "missing_policy": "nonfinite and negative discharge are invalid; zero is retained",
  "observation_model_unit": "mm/day",
  "observation_raw_unit": "ft3/s",
  "observation_source": "/home/jingxin/code/dmg-research/data/camels_dataset",
  "online_statistics_inputs": [
    "/home/jingxin/code/dmg-research/project/hydro_structure_diagnosis/manuscript/results/R1/r1_online_performance.csv",
    "/home/jingxin/code/dmg-research/project/hydro_structure_diagnosis/manuscript/results/R1/r1_online_signature_basin_year.csv"
  ],
  "periods": {
    "test": {
      "days": 5479,
      "end": "2010-09-30",
      "end_index": 10956,
      "name": "test",
      "start": "1995-10-01",
      "start_index": 5478
    },
    "test_forcing": {
      "days": 5844,
      "end_index": 10957,
      "preceding_warmup_days": 365,
      "start_index": 5113
    },
    "test_target_days": 5479,
    "train": {
      "days": 5113,
      "end": "1995-09-30",
      "end_index": 5477,
      "name": "train",
      "start": "1981-10-01",
      "start_index": 365
    },
    "train_forcing": {
      "days": 5478,
      "end_index": 5478,
      "start_index": 0
    },
    "train_target_days": 5113,
    "warmup": {
      "days": 365,
      "end": "1981-09-30",
      "end_index": 364,
      "name": "warmup",
      "start": "1980-10-01",
      "start_index": 0
    }
  },
  "selected_tgd2": {
    "epoch": 100,
    "root": "/home/jingxin/code/dmg-research/project/hydro_structure_diagnosis/results/dpl_camels_531_lite_v3_tgd2_dpl_audited/XAJ_TGD2",
    "rule": "maximum checkpoint epoch common to all three seeds and present in epoch_history",
    "seeds": [
      {
        "checkpoint": "/home/jingxin/code/dmg-research/project/hydro_structure_diagnosis/results/dpl_camels_531_lite_v3_tgd2_dpl_audited/XAJ_TGD2/seed_42/checkpoint_epoch_100.pt",
        "config": "/home/jingxin/code/dmg-research/project/hydro_structure_diagnosis/results/dpl_camels_531_lite_v3_tgd2_dpl_audited/XAJ_TGD2/seed_42/config.json",
        "epoch": 100,
        "seed": "42",
        "timestamp": "2026-07-30T12:46:57.559767008+00:00"
      },
      {
        "checkpoint": "/home/jingxin/code/dmg-research/project/hydro_structure_diagnosis/results/dpl_camels_531_lite_v3_tgd2_dpl_audited/XAJ_TGD2/seed_123/checkpoint_epoch_100.pt",
        "config": "/home/jingxin/code/dmg-research/project/hydro_structure_diagnosis/results/dpl_camels_531_lite_v3_tgd2_dpl_audited/XAJ_TGD2/seed_123/config.json",
        "epoch": 100,
        "seed": "123",
        "timestamp": "2026-07-30T12:39:29.975836039+00:00"
      },
      {
        "checkpoint": "/home/jingxin/code/dmg-research/project/hydro_structure_diagnosis/results/dpl_camels_531_lite_v3_tgd2_dpl_audited/XAJ_TGD2/seed_2026/checkpoint_epoch_100.pt",
        "config": "/home/jingxin/code/dmg-research/project/hydro_structure_diagnosis/results/dpl_camels_531_lite_v3_tgd2_dpl_audited/XAJ_TGD2/seed_2026/config.json",
        "epoch": 100,
        "seed": "2026",
        "timestamp": "2026-07-30T12:42:39.471810102+00:00"
      }
    ]
  },
  "status": "complete"
}
