# Figure 4 Candidate A Caption Draft

**Figure 4 (Candidate A). Parameter-space occupation and global shift patterns between Base and CN model structures.**

**(a) Global signed paired parameter shifts ($\Delta z = z_{\text{Base}} - z_{\text{CN}}$).** Medians and 95% bootstrap confidence intervals (10,000 resamples across 531 basins) for all 15 shared parameters in normalized $[0,1]$ physical space. Results for independent calibration (IC, CMA-ES 10 restarts, warm orange circle) and differentiable parameter learning (dPL, neural parameter network 3 seeds, deep blue square) are shown side-by-side. The dashed vertical reference line indicates zero shift.

**(b) IC normalized parameter-space occupation.** Actual parameter distributions for Base (warm orange) and CN (deep blue) model structures across 531 basins under IC. For each parameter row, white markers represent medians, thick bars denote interquartile ranges (IQR, 25th–75th percentiles), and thin whiskers indicate 5th–95th percentile ranges. All parameters are displayed on a common normalized scale $z \in [0, 1]$.

**(c) dPL normalized parameter-space occupation.** Actual parameter distributions for Base (warm orange) and CN (deep blue) model structures across 531 basins under dPL, displayed using the same interval representation ($z \in [0, 1]$) as in panel (b).

*Note: Parameter values ($z$) are linearly mapped to $[0,1]$ based on audited physical bounds. IC and dPL represent distinct parameter-estimation paradigms; distribution shifts reflect how model structures occupy parameter space under different learning constraints rather than optimization efficiency or physical truth.*
