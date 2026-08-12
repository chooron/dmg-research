# Configuration catalog

- `dpl_hbv_kgeq_365d_v1.json`: historical HBV window-ablation configuration,
  retained because the maintained HBV analysis entry point references it.
- `ic_xnes_production_v1.json`: legacy 559-basin XNES configuration used by
  historical reproduction/analysis tools. It contains retired old-TGD model
  keys and is not an active production configuration.

New formal configurations should use the current 531-basin protocol, be
human-authored here only when reusable, and be copied as a fully resolved
config into the corresponding `results/<run_id>/` directory.

