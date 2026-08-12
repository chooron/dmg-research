# HBV DPL KGE(Q) window-length ablation plan

## Objective

Select the prediction-window length for the static-attribute-to-HBV-parameter
DPL before changing capacity, parameter supervision, or loss definitions.
The DPL is trained end-to-end against KGE(Q); no IC parameter labels are used
in this ablation.

## Fixed protocol

- Basins: the same 559 CAMELS basins used by IC v4.
- Calibration: 1989-01-01 to 1998-12-31.
- Evaluation: 1999-01-01 to 2009-12-31, with the preceding 365 days as
  evaluation warmup.
- Input: 35 static CAMELS attributes, median/IQR normalized and clipped to
  [-5, 5].
- Parameterizer: `35 -> 64 -> 64 -> 12`, LayerNorm, SiLU, dropout 0.05,
  sigmoid output in normalized parameter space, initialized at HBV physical
  defaults.
- Optimizer: AdamW, learning rate `1e-2`, weight decay `1e-4`, gradient clip
  1.0, cosine decay to `1e-4`.
- Training precision: FP32; final evaluation: FP64 model rerun and KGE.
- Objective: mean of basin-wise KGE(Q) losses, not pooled KGE.
- Epochs and seed for screening: 100 epochs and seed 42.

## Screening runs (strictly sequential)

| Order | Warmup | Prediction | Windows/epoch | Predicted calibration days/epoch |
|---:|---:|---:|---:|---:|
| A1 | 365 | 365 | 10 | 3650 |
| A2 | 365 | 730 | 5 | 3650 |
| A3 | 365 | 1825 | 2 | 3650 |

The 3652-day calibration period leaves the same final two days unused in all
three screening runs. Therefore the screening comparison changes the temporal
credit-assignment horizon, not the number of supervised days per epoch.

Run A2 only after A1 has produced `COMPLETE`; run A3 only after A2. This avoids
contention on the single local GPU and keeps the logs easy to audit.

## Selection rule

Choose the window with the highest median FP64 evaluation KGE(Q). Break ties
within 0.005 using, in order: higher mean evaluation KGE, fewer non-finite
forwards, and lower epoch-to-epoch validation variation.

## Confirmation and later ablations

1. Re-run the selected window for 200 epochs with seeds 42, 123, and 2026.
2. Only after window selection, test learning rate `3e-3` and `1e-3` against
   the current `1e-2` setting.
3. Separately test IC-parameter Huber pretraining followed by end-to-end DPL
   fine-tuning. It must not be mixed into the window-length comparison.
4. Compare the selected deterministic network against a process-group output
   head only if the compact single-head baseline is performance-limited.
