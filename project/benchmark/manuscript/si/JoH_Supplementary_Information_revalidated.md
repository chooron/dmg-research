# Supplementary Information: Differentiable Parameter Learning Across 36 Conceptual Hydrological Models

**Journal of Hydrology**  
*Supplementary Information for: "Structural Constraints, Parameter Space Organization, and Generalization Mechanics in Differentiable Hydrological Modeling"*

---

## S1. Differentiable Model Reconstruction and Numerical Verification

### S1.1 Differentiable Model Reconstruction
To evaluate parameter space geometry and information learning across diverse conceptual architectures, 36 lumped hydrological model structures derived from the MARRMoT framework (Westerberg et al., 2020; Knoben et al., 2019) were reconstructed within an end-to-end differentiable computational graph implemented in PyTorch (`dmotpy`). The model portfolio spans 1 to 15 calibrated parameters (271 parameters in total across the ensemble) and 1 to 6 dynamical storage stores (110 total store configurations; 96 unique dynamic stores).

Each model reconstructs the physical flux topology and storage mass balance of its conceptual parent while replacing non-differentiable operations with smooth, autograd-compatible formulations:

1. **State Update and Flux Formulation**: State evolution follows explicit mass-conserving discrete-time storage updates:
   $$\mathbf{S}_{t+1} = \mathbf{S}_t + \Delta t \sum \mathbf{F}_{\text{in}}(\mathbf{S}_t, \mathbf{x}_t, \boldsymbol{\theta}) - \Delta t \sum \mathbf{F}_{\text{out}}(\mathbf{S}_t, \mathbf{x}_t, \boldsymbol{\theta})$$
   where $\mathbf{S}_t$ is the vector of catchment storages, $\mathbf{x}_t$ represents daily meteorological forcing inputs, $\boldsymbol{\theta}$ denotes the physical parameter vector, and $\Delta t = 1.0\text{ day}$. Storage bounds are maintained via mass-conserving flux allocation where outflow demands are capped by available storage: $\mathbf{F}_{\text{actual}} = \min(\mathbf{F}_{\text{demand}}, \mathbf{S}_t / \Delta t)$.

2. **Continuous Gating and Smoothing Operators**: Discontinuous threshold functions (such as Heaviside step functions governing field capacity overflow, saturation excess, and temperature-dependent snowmelt switches) are reformulated using scaled logistic sigmoids:
   $$\sigma_k(u) = \frac{1}{1 + \exp\left(-\frac{k}{\tau} u\right)}$$
   where $u = S - S_{\text{thresh}}$ or $u = T - T_{\text{melt}}$, and $k/\tau$ is a sharpness parameter chosen to provide steep threshold transitions while preserving non-vanishing local gradients for backpropagation. Non-smooth flux caps and piecewise linear functions are regularized using smooth softplus relaxations ($\text{smooth\_relu}(x, \beta) = \frac{1}{\beta} \ln(1 + \exp(\beta x))$) and smooth minimum functions ($\text{smooth\_min}(a, b, \beta) = a - \text{smooth\_relu}(a - b, \beta)$).

3. **Numerical Safeguards**: Zero-division hazards in empirical flux equations (e.g., non-linear routing and exponential infiltration formulations) are stabilized using machine-precision floor offsets:
   $$\text{safe\_div}(a, b) = \frac{a}{b + \epsilon}, \quad \epsilon = 1.0 \times 10^{-6}$$

4. **Differentiable Unit Hydrograph Routing**: Channel and catchment delay routing is implemented through 9 differentiable unit hydrograph convolution kernels (`uh_0` to `uh_8`), covering half-triangular, full-triangular, exponential, gamma, and uniform distributions. Convolution operations are computed via 1D causal convolutions over runoff time series.

### S1.2 Numerical Verification Suite
The 36 reconstructed differentiable models underwent a comprehensive four-tier verification protocol:

1. **Water Balance Closure**: Mass conservation was evaluated across 12 synthetic and historical climate regimes under double-precision (`float64`) and single-precision (`float32`) arithmetic. Across the 36 models, 35 achieve strict water balance closure with maximum absolute daily storage residuals below $1.0 \times 10^{-3}\text{ mm/day}$ (typical double-precision residuals $\sim 10^{-11}\text{ mm/day}$). One model (`vic`) exhibits a minor boundary clipping residual ($0.274\text{ mm/day}$ in `float64`, $0.406\text{ mm/day}$ in `float32`) under extreme synthetic stress tests due to hard store boundary clamping.
2. **Forward Numerical Stability**: 100% of the 36 models run forward without encountering non-finite outputs (`NaN` or `Inf`) across all 531 CAMELS-US catchments and synthetic boundary forcing datasets.
3. **Gradient Correctness and Autograd Audit**: End-to-end gradient correctness was verified by backpropagating loss gradients through the 15-year simulation horizon. All 36 models produce valid non-zero gradient vectors. Thirteen representative models spanning distinct routing topologies, snow routines, and multi-layer soil cascades were subjected to double-precision finite-difference gradient checks (`torch.autograd.gradcheck`), passing with analytical-to-numerical gradient tolerances $< 1.0 \times 10^{-4}$. Gradient sparsity audits identified 7 models (`alpine2`, `gr4j`, `hbv96`, `modhydrolog`, `newzealand2`, `plateau`, `smar`) that exhibit zero gradients for inactive parameter subsets under dry, snow-free, or sub-threshold forcing regimes (e.g., snowmelt degree-day factors in warm catchments).
4. **Time-Step and Euler Convergence**: Substep convergence was tested across Euler discretization levels ($K = 1, 2, 4, 8$ substeps per day). Twenty-three models exhibit nominal first-order convergence rates ($p \in [0.85, 1.15]$); 3 models (`hymod`, `mopex1`, `vic`) recover first-order scaling at finer substeps ($K=4$); 18 models display threshold-dominated non-smooth convergence behavior characteristic of conceptual bucket thresholding; and 1 model (`wetland`) reaches the numerical precision floor.

Table S1 (revalidated) summarizes the verification status and architecture of all 36 models.

### S1.3 Performance-Context Comparison with Reference MARRMoT Calibrations
To confirm that the reconstructed differentiable ensemble occupies an expected performance range under standard hydrological calibration, the 36 differentiable models were compared against historical reference calibrations from the original MARRMoT framework across overlapping CAMELS-US catchments (Level B comparison). Both model suites were calibrated and evaluated across identical 15-year historical periods:
- **Parameter Estimation Period**: 1 October 1980 to 30 September 1995 (5,478 days + 5-year warm-up)
- **Evaluation Period**: 1 October 1995 to 30 September 2010 (5,479 days)

Across all 36 models and overlapping catchments ($N = 17,380$ evaluation pairs, $480\text{--}483$ catchments per model):
1. **Model-Level Performance Ranking**: Median evaluation KGE across the 36 differentiable models strongly tracks median performance in the reference MARRMoT suite, achieving a Spearman rank correlation of $\mathbf{\rho = 0.7079}$ ($p = 1.38 \times 10^{-6}$) over the evaluation period and $\mathbf{\rho = 0.7385}$ ($p = 2.73 \times 10^{-7}$) over the calibration period. The model-level median difference ($\text{KGE}_{\text{DMOT}} - \text{KGE}_{\text{MARRMoT}}$) is $+0.0324$ ($\text{IQR} = 0.0575$) in evaluation and $+0.0329$ ($\text{IQR} = 0.0650$) in calibration.
2. **Catchment-Level Efficiency Distribution**: At the individual catchment level, the median paired evaluation difference is $\Delta\text{KGE} = \mathbf{+0.0274}$ ($\text{IQR} = 0.1104$, $\text{Q25} = -0.0137, \text{Q75} = +0.0967$), with the differentiable implementation achieving comparable or higher evaluation efficiency in $69.57\%$ of catchment instances ($77.74\%$ in calibration).

*Methodological Distinction*: This analysis represents a macro-level performance-context comparison under identical temporal and forcing conditions, rather than a parameter-for-parameter implementation identity test. Exact parameter transfer is precluded by fundamental solver differences (MATLAB MARRMoT uses continuous adaptive ODE solvers with discontinuous switches, whereas the differentiable models use discrete Euler time-stepping with smooth sigmoidal transitions). The high rank correlation and close median agreement confirm that the reconstructed models preserve the relative structural capabilities and hydrologic efficacy of the parent MARRMoT suite.

Manuscript Figure S1 presents the complete $36 \times 531$ empirical matrix of raw performance differences ($\Delta\text{KGE} = \text{KGE}_{\text{dPL}} - \text{KGE}_{\text{IC}}$) across all $19,116$ model–basin cases.

---

## S2. Data and Parameter-Estimation Configuration

### S2.1 Catchment Dataset and Preprocessing
Experiments were conducted across the 531-catchment benchmark subset of the CAMELS-US dataset (Addor et al., 2017; Newman et al., 2015). Daily meteorological forcing inputs were extracted from Daymet, including daily total precipitation ($P$, mm/day), daily mean air temperature ($T$, °C), and daily potential evapotranspiration ($\text{PET}$, mm/day) calculated using the Hargreaves method. Observed streamflow records ($\text{ft}^3/\text{s}$) were converted to area-normalized runoff ($Q$, mm/day) using catchment drainage areas (`area_gages2`). Missing or invalid streamflow observations were preserved as `NaN` and masked out of objective function and evaluation metrics via boolean validity masks.

The temporal record spans 30 hydrological years partitioned into two non-overlapping periods:
- **Parameter Estimation (Training / Calibration)**: 1 October 1980 to 30 September 1995 (15 hydrological years = 5,478 daily timesteps).
- **Evaluation (Testing)**: 1 October 1995 to 30 September 2010 (15 hydrological years = 5,479 daily timesteps).

For evaluation runs across both individual calibration and dPL, a 365-day warm-up period (1 October 1994 to 30 September 1995) was prepended to establish dynamical storage states; warm-up timesteps were strictly detached from evaluation metric calculations.

### S2.2 Model-Specific Temporal Exceptions
Three models require calendar Day of Year (DOY, $1 \dots 366$) as an auxiliary fourth input channel:
1. `mopex4` and `mopex5`: utilize DOY in seasonal cosine formulations governing annual vegetation interception dynamics.
2. `vic`: uses DOY in empirical phenology curves to modulate seasonal dynamic canopy capacity.

`penman` utilizes a 365-day warm-up window (730-day total window = 365-day warm-up + 365-day scored period) rather than the 730-day warm-up used for the remaining 35 models during dPL mini-batch training. Warm-up states are detached across all 36 models (`"detach"` autograd mode).

### S2.3 Catchment Physical Attributes and Information Space
The seen-basin dPL neural parameterizer ingests 35 static CAMELS-US physical attributes spanning climate (9), topography (3), vegetation (7), soils (9), and geology (7). Prior to neural parameterization:
1. Highly skewed attributes (`area_gages2`, `soil_conductivity`, `geol_permeability`) undergo logarithmic transformation ($\ln(x + \text{shift})$).
2. All 35 attributes are standardized to zero mean and unit variance ($Z$-score normalization) across the training catchments.

To analyze catchment-to-parameter relationships without collinearity redundancy, the 35 attributes were grouped using average-linkage hierarchical clustering on absolute Spearman distance ($d(a_i, a_j) = 1 - |\rho_{\text{Spearman}}(a_i, a_j)|$) at a cutoff threshold of $0.30$ (corresponding to $|\rho| \ge 0.70$). Multi-attribute clusters are represented by their first principal component score ($PC1$), which captures $52.2\%$ to $95.1\%$ of intra-cluster variance across the 531 catchments. This partitions the 35 attributes into 20 orthogonal seen-basin information dimensions (Table S2).

In the Out-of-Bag (OOB) generalization experiments, 3 discrete categorical codes (`dom_land_cover`, `geol_1st_class`, `geol_2nd_class`) were excluded to avoid arbitrary ordinal assumptions, leaving 32 continuous attributes. Connected-component clustering at $|\rho| \ge 0.70$ groups these 32 continuous attributes into 13 orthogonal physical clusters (Table S2).

### S2.4 Parameter Estimation Configurations
Table S3 provides a side-by-side comparison of the configuration settings for Individual Calibration (IC), Seen-Basin dPL, and Held-Out (OOB) dPL.

#### Individual Calibration (IC)
Individual calibration optimizes parameter vectors independently for each model–basin pair ($36 \times 531 = 19,116$ calibration tasks) using Batched Active CMA-ES implemented in PyTorch under full double precision (`float64`). For each task, 10 independent optimization restarts were initialized via Latin Hypercube Sampling in the logit latent space ($z_0 \in \mathbb{R}^D$, $\sigma_0 = 0.10$). Each restart was optimized for 300 generations (generation 280 for `simhyd`) using a tiered population size ($\lambda = 8$ for $D=1$; $\lambda = 12$ for $D \in [4, 6]$; $\lambda = 16$ for $D \in [7, 10]$; $\lambda = 20$ for $D \in [12, 15]$). A 5-repeat 1-year warm-up cycle ($5 \times 365 = 1,825$ days) was prepended prior to the calibration period. The canonical IC solution was selected as the restart achieving the highest Kling-Gupta Efficiency ($\text{KGE}$) over the 1980–1995 calibration period.

#### Differentiable Parameter Learning (dPL)
Seen-basin dPL trains a single shared catchment parameterizer ($\mathcal{M}_\phi$) across all 531 basins simultaneously. The parameterizer is a Multi-Layer Perceptron (MLP) with two hidden layers of 256 units each, Layer Normalization, GELU activations, and a 5% dropout rate. The output layer is initialized with zero weights and zero biases, setting initial normalized outputs strictly to parameter interval midpoints ($u_0 = 0.5$).

The network is trained using AdamW ($\text{lr} = 1.0 \times 10^{-3}$, weight decay $= 1.0 \times 10^{-4}$, gradient norm clipping at $1.0$) for a maximum of 100 epochs with mini-batches of 100 catchments randomly sampled across 169 steps per epoch ($16,900$ catchment-window samples per epoch). Sequence horizons comprise 1095 days (730 days detached warm-up + 365 days scored simulation; 365+365 days for `penman`). The training objective minimizes mean ensemble error:
$$\mathcal{L}(\phi) = \frac{1}{B} \sum_{b=1}^B \left(1 - \text{KGE}_b(Q_{\text{obs}}, Q_{\text{sim}}(\mathcal{M}_\phi(\mathbf{a}_b)))\right)$$
Early stopping monitors training loss with a patience of 10 epochs and a minimum threshold of 50 epochs. The model checkpoint achieving the minimum training loss (`best.pt`) is reloaded for all subsequent evaluations.

#### Parameter Bound Mapping
Normalized parameter coordinates $u \in (0, 1)$ generated by the sigmoid output layer are mapped to physical parameter ranges $[\theta_{\min}, \theta_{\max}]$:
- **Linear Mapping**: If the lower bound is non-positive or the dynamic range is narrow ($\theta_{\max} / \theta_{\min} < 100$):
  $$\theta = \theta_{\min} + u \cdot (\theta_{\max} - \theta_{\min})$$
- **Logarithmic Mapping**: If $\theta_{\min} > 0$ and $\theta_{\max} / \theta_{\min} \ge 100$:
  $$\theta = \exp\left(\ln(\theta_{\min}) + u \cdot (\ln(\theta_{\max}) - \ln(\theta_{\min}))\right)$$

Across the 271 calibrated parameters in the 36-model ensemble, exactly **53 parameters** ($19.56\%$) use logarithmic mapping and **218 parameters** ($80.44\%$) use linear mapping (complete inventory in Table S3).

---

## S3. Parameter-Space, Information-Organization, and Held-Out-Basin Analyses

### S3.1 Parameter-Space Analysis Mechanics
To analyze geometric transformations between IC and dPL parameterizations, all parameter values are projected into the dimensionless normalized coordinate hypercube $\tilde{\theta} \in [0, 1]^{P_m}$:
$$\tilde{\theta}(m, b, p) = \frac{\theta(m, b, p) - \theta_{\min}(m, p)}{\theta_{\max}(m, p) - \theta_{\min}(m, p)} \quad (\text{or } \tilde{\theta}_{\log} \text{ for log-mapped coordinates})$$

1. **Parameter Vector Displacement ($D_{\text{RMS}}$)**: Root-mean-square coordinate distance between IC and dPL parameterizations across catchment $b$:
   $$D_{\text{RMS}}(m, b) = \sqrt{\frac{1}{P_m} \sum_{p=1}^{P_m} \left(\tilde{\theta}_{\text{IC}}(m, b, p) - \tilde{\theta}_{\text{dPL}}(m, b, p)\right)^2}$$
   Across all 36 models and 531 catchments ($N = 19,116$ pairs), the grand model-equal median displacement is $D_{\text{RMS}} = \mathbf{0.384}$.

2. **IC-Self Reference and Separation Excess**: To test whether $D_{\text{RMS}}$ exceeds the dispersion inherent in equifinal IC calibrations, an IC-self reference dispersion $D_{\text{self}}(m, b)$ was computed from the 10-restart CMA-ES archive. A restart was deemed eligible if its calibration KGE satisfied $\text{KGE}_{\text{restart}} \ge \text{KGE}_{\text{best}} - 0.01$. Twenty-three models achieved $\ge 90\%$ catchment coverage of valid non-canonical restarts and formed the primary IC-self benchmark pool (13 models had insufficient near-optimal restarts due to sharp global optima). The model-equal paired excess displacement is:
   $$E_{\text{paired}} = \text{median}_m \left(\text{median}_b \left(D_{\text{cross}}(m, b) - D_{\text{self}}(m, b)\right)\right) = \mathbf{+0.2188}$$
   with a 95% bootstrap confidence interval of $[0.2091, 0.2281]$. Across all 36 models, $100\%$ ($36/36$) exhibit positive median separation excess ($87.58\%$ of all $19,116$ catchment pairs).

3. **Coordinate Localization and Participation Ratio**: Coordinate-wise shift magnitude is defined as $M_{\text{cross}}(m, p) = \text{median}_b |\tilde{\theta}_{\text{IC}}(m, b, p) - \tilde{\theta}_{\text{dPL}}(m, b, p)|$. The effective parameter participation ratio is:
   $$C_{\text{eff}} = \frac{\left(\sum_p M_{\text{cross}}(m, p)^2\right)^2}{P_m \sum_p M_{\text{cross}}(m, p)^4}$$
   Across the 36 models, raw $C_{\text{eff}} = 0.3458$, with the single most displaced coordinate accounting for $54.56\%$ of total squared displacement and the top-2 coordinates accounting for $84.82\%$. When adjusting for IC-self baseline dispersion across the strict 23-model pool, adjusted $C_{\text{eff}} = 0.2934$, with top-1 and top-2 shares rising to $67.26\%$ and $94.18\%$, demonstrating that dPL parameter adjustments localize onto 1–2 dominant coordinates rather than dispersing isotropically.

4. **Contraction Ratio Sensitivity**: The coordinate contraction ratio $\text{CR}(m, p) = \text{IQR}_b(\tilde{\theta}_{\text{dPL}}(m, b, p)) / \text{IQR}_b(\tilde{\theta}_{\text{IC}}(m, b, p))$ exhibits strong dependence on the IC baseline definition. While comparing dPL against a single canonical IC seed produces an apparent median contraction of $\text{CR}_{\text{canonical}} = 0.6140$, comparing against consensus multi-start IC medians yields $\text{CR}_{\text{consensus}} = 0.9795$, and comparing against synthetic IC-self realizations yields $\text{CR}_{\text{selfref}} = 1.0043$. Apparent parameter variance contraction under dPL is thus primarily an artifact of single-seed IC reference selection.

5. **Performance–Parameter Bridge**: The within-model Spearman rank correlation between performance gap magnitude ($|\Delta\text{KGE}| = |\text{KGE}_{\text{dPL}} - \text{KGE}_{\text{IC}}|$) and parameter displacement ($D_{\text{RMS}}$) yields a model-equal median of $\rho_b = \mathbf{+0.2412}$ (95% CI: $[0.1973, 0.2586]$), with $34/36$ models ($94.4\%$) showing positive rank associations.

### S3.2 Information-Organization and Reproducibility Metrics
The complete information space consists of $271\text{ parameter coordinates} \times 20\text{ seen information dimensions} = \mathbf{5,420\text{ primary cells}}$.

1. **IC-Stable Denominator**: To avoid false-positive associations driven by unidentifiable or flat parameter directions, an IC-stability gate was applied strictly to IC calibration results:
   $$|\rho_{\text{IC}}| \ge 0.20 \quad \text{AND} \quad P(\text{sign bootstrap}) \ge 0.95$$
   Exactly **902 cells** ($16.64\%$ of the 5,420 population) satisfy this stability criterion.

2. **Sign and Magnitude Retention**: Among the 902 IC-stable cells, dPL retains the identical correlation sign in **849 cells** ($\mathbf{94.12\%}$). Magnitude retention rates at increasingly stringent dPL thresholds are:
   - Same sign and $|\rho_{\text{dPL}}| \ge 0.10$: $\mathbf{87.69\%}$ ($791 / 902$)
   - Same sign and $|\rho_{\text{dPL}}| \ge 0.20$: $\mathbf{76.72\%}$ ($692 / 902$)
   - Same sign and $|\rho_{\text{dPL}}| \ge 0.30$: $\mathbf{57.32\%}$ ($517 / 902$)

3. **Four-Way Classification for Strong dPL Relationships**: When evaluating all strong relationships in dPL ($|\rho_{\text{dPL}}| \ge 0.20$), cells are partitioned into:
   - `IC-supported`: $|\rho_{\text{IC}}| \ge 0.20$ and $\text{sign}(\rho_{\text{IC}}) = \text{sign}(\rho_{\text{dPL}})$ ($33.32\%$).
   - `IC-opposite-sign`: $|\rho_{\text{IC}}| \ge 0.20$ and $\text{sign}(\rho_{\text{IC}}) \ne \text{sign}(\rho_{\text{dPL}})$.
   - `IC-weak`: $0.10 \le |\rho_{\text{IC}}| < 0.20$.
   - `IC-absent`: $|\rho_{\text{IC}}| < 0.10$.
   Cells in the latter three categories are classified as `dPL-emergent` ($66.68\%$).

4. **Paired Profile Correspondence and Diagonal Advantage**: For each parameter coordinate $p$ in model $m$, paired profile correspondence between IC and dPL is measured across the 20 information dimensions:
   $$R_{m, p} = \text{Spearman}_k(\rho_{\text{IC}}[m, p, k], \rho_{\text{dPL}}[m, p, k]), \quad R_{\text{paired}} = \text{median}_m(\text{median}_p(R_{m, p})) = \mathbf{0.7158}$$
   Within-model profile cross-correlation matrices ($C_m[p, q]$) show that diagonal profile similarity (median $0.7158$) substantially exceeds off-diagonal cross-parameter similarity (median $-0.0218$), yielding a model-equal diagonal advantage of:
   $$A_{\text{diag}} = \text{median}_m \left(\text{median}_p C_m[p, p] - \text{median}_{q \ne p} C_m[p, q]\right) = \mathbf{0.6150}$$
   Across all 35 multi-parameter models ($P \ge 2$), $100\%$ ($35/35$) exhibit $A_{\text{diag}} > 0$. A 1,000-replicate parameter label permutation test yields a null mean of $0.0015$ (95% null interval: $[-0.1233, 0.1128]$, empirical $p = \mathbf{0.000999}$).

5. **Functional-Role Negative Control**: To test whether profile correspondence reflects abstract hydrological functional roles (e.g., fast routing vs. baseflow) rather than exact mathematical coordinate identities, same-role off-diagonal correlations were contrasted against cross-role correlations. Across 29 models with valid functional role pairings, role advantage is negative ($A_{\text{role}} = -0.1188$, permutation $p = 0.8057$), confirming that information transfer is strictly coordinate-specific.

### S3.3 Out-of-Bag (OOB) Generalization Protocol
To test whether parameter organization and information structures persist in unseen basins, an Out-of-Bag (OOB) cross-validation experiment was executed across a prespecified panel of 8 representative models (`alpine2`, `hbv96`, `xinanjiang`, `newzealand2`, `ihacres`, `us1`, `mopex4`, `hillslope`).

1. **Pre-OOB Selection Rule**: Models were selected deterministically prior to OOB execution by partitioning the 36-model seen benchmark into 4 orthogonal quadrants spanning flexibility gap ($G_{\text{seen}} = \text{median}_b(\text{KGE}_{\text{IC}} - \text{KGE}_{\text{dPL}})$) and parameter reproducibility ($R_{\text{seen}} = R_{\text{paired}}$), enforcing a baseline performance viability gate ($K_{\text{joint}} \ge 0.561$), maximizing 5D multidimensional dispersion, and revising MOPEX structural redundancy.
2. **5-Fold Cross-Validation and Assembly**: Catchments were partitioned into 5 disjoint folds using deterministic spatial/random cross-validation (`KFold`, seed `20260902`, fold sizes: $107, 106, 106, 106, 106$). Forty formal training jobs ($8 \text{ models} \times 5 \text{ folds} = 40$) were executed. In each fold, attribute normalization statistics were fitted strictly on training catchments, and streamflow observations for held-out catchments were strictly excluded from parameter estimation. Out-of-fold predictions were concatenated across the 5 folds to yield complete 531-basin uncalibrated parameterizations.
3. **Replay Findings**:
   - *Outlet Performance Transfer*: Median $\Delta\text{KGE}_{\text{OOB}} = -0.0654$. Two-way ANOVA decomposes performance variation into $67.14\%$ catchment characteristics, $32.33\%$ remainder, and $0.53\%$ model structure.
   - *Parameter Displacement*: $D_{\text{RMS, OOB}} = \mathbf{0.3429}$, showing minimal shift relative to seen-dPL parameterizations (mean parameter shift between seen-dPL and OOB-dPL is only $0.1004$).
   - *Information Space Persistence*: Profile correspondence between IC and OOB-dPL is $R_{\text{paired, OOB}} = \mathbf{0.7390}$ across 13 continuous clusters ($0.7620$ across 32 continuous attributes), with diagonal advantage $A_{\text{diag, OOB}} = \mathbf{0.7184}$ ($p = 0.000999$).
   - *Direct dPL-to-dPL Consistency*: Direct correlation between seen-dPL and OOB-dPL relationship matrices across all parameter-attribute pairs reaches $\rho = \mathbf{0.9616}$ ($99.93\%$ sign retention; $96.44\%$ cluster retention), confirming that the learned parameter-attribute mapping is an intrinsic regionalized mapping that generalizes robustly to ungauged catchments.

Table S4 provides the formal estimand, population, aggregation hierarchy, and OOB replay status for all study metrics.

---

## References
1. Addor, N., Newman, A. J., Mizukami, N., & Clark, M. P. (2017). The CAMELS data set: catchment attributes and meteorology for large-sample studies. *Hydrology and Earth System Sciences*, 21(10), 5293–5313.
2. Knoben, W. J., Freer, J. E., Fowler, K. J., Peel, M. C., & Woods, R. A. (2019). Modular Assessment of Rainfall–Runoff Models Toolbox (MARRMoT) v1. 2: an open-source, extendable framework providing implementations of 46 conceptual hydrologic models as continuous state-space formulations. *Geoscientific Model Development*, 12(6), 2463–2480.
3. Newman, A. J., Clark, M. P., Sampson, K., Wood, A., Hay, L. E., Bock, A., ... & Duan, Q. (2015). Development of a large-sample watershed-scale hydrometeorological data set for the contiguous USA: data set characteristics and assessment of regional variability in hydrologic model performance. *Hydrology and Earth System Sciences*, 19(1), 209–223.
4. Westerberg, I. K., Sikorska-Senoner, A. E., Vivroli, D., Vis, M., & Seibert, J. (2020). Hydrological model evaluation with uncertain data: a review. *Hydrological Sciences Journal*, 65(13), 2215–2235.
