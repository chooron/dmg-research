# Figure S2 provenance

- **Scientific question:** Does the separation between outlet recovery and seasonal internal/pathway recovery localize to the high-snow seasonal cycle in the controlled R3 experiment?
- **Input files:** `manuscript/results/R3/fig6_seasonal/fig6_seasonal_input.npz`, `fig6_seasonal_state.npz`, `fig6_seasonal_meta.json`, and `manuscript/results/R3/figure6_summary.json`.
- **Input keys:** Six regime series (`Base_IC`, `Base_dPL`, `TGD2_IC`, `TGD2_dPL`, `CN_IC`, `CN_dPL`) in each NPZ; metadata records `basin_ids`, the effective-input quantity, `wt = wu + wl + wd`, and water-year months.
- **Subset and N:** `f_snow >= Q75`; recorded threshold `0.21769666937653748`; **N=133** high-snow basins; test period 1995-10-01 to 2010-09-30.
- **Aggregation:** Per-basin mean for each water-year month; dPL is median across seeds per basin; panel-level summaries use the basin median and recorded interval fields.
- **IC/dPL handling:** IC and dPL are shown as separate line styles for Base, TGD, and CN; they are not pooled.
- **Interval definition:** The plotting script uses `ci_lo`/`ci_hi`, documented by `prepare_figure6_data.py` as 2,000-draw catchment-resampling 95% CIs of the monthly median. The metadata string saying “median and IQR” is inconsistent and should be corrected in the formal Supplement text.
- **Plot script:** `manuscript/scripts/r3/plot_r3_si_seasonal_trajectories.py`.
- **Output:** Source `manuscript/figures/Fig_S6_R3_seasonal_trajectories.png`; final submission copy `manuscript/supplement/figures/FigureS2_R3_seasonal_trajectories.png`.
- **Canonical values checked:** Metadata records six `(133, 12)` output shapes and the effective liquid-water input/common XAJ storage definitions. The source PNG was copied byte-for-byte; its old S6 filename is retained.
