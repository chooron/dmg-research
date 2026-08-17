# Flex-MOPEX Final Method Specification (R19 Canonical Freeze)

## 1. Executive Summary

This document specifies the canonical, frozen **Flex-MOPEX Counterfactual Structural Learning** architecture and training protocol established at Round 19 (R19). 

The method resolves the longstanding structural gate collapse and representation entanglement problem in differentiable hydrological models. It achieves:
1. **Four-Process Structural Polarization**: Simultaneously polarizes all four structural gates (`w_phen`, `w_int`, `w_snow`, `w_sub`) with positive separation ($\Delta > 0$) and positive rank correlation with continuous physical ground-truth across multiple independent random seeds.
2. **Predictive Performance Record**: Achieves an all-time record 5114-day median NSE of **0.6502 ± 0.0012** across seeds (peak 0.6518 on Seed 42), outperforming standard differentiable baseline models (+0.0185 NSE) while maintaining strict physical interpretability.
3. **Engineering Simplicity**: Uses a **single unified Adadelta optimizer** across all parameters with zero hyperparameter tuning.

---

## 2. Model Architecture

### 2.1 Hydrological Physics (`LearnedWeightMopexE`)
- **Interception formulation**: Candidate E-S0 (smooth continuous interception canopy storage with explicit drainage/evaporation semantics).
- **Process gates (4 processes, 2 logits each)**:
  - `w_phen`: Vegetation phenology dynamic LAI control ($p=0$, cost $c_p=2.0$)
  - `w_int`: Canopy interception storage & evaporation ($p=1$, cost $c_p=2.0$)
  - `w_snow`: Snow accumulation & temperature-index melt ($p=2$, cost $c_p=2.0$)
  - `w_sub`: Groundwater / baseflow slow routing storage ($p=3$, cost $c_p=1.0$)
- **Gate activation**: $w_p = \text{Softmax}(\text{logits}_p)[1] \in (0, 1)$.

### 2.2 Hydrological Parameter & Representation Backbone
- **Input**: 35 normalized static basin physical attributes $x_{35} \in \mathbb{R}^B \times 35$.
- **Backbone**: Linear(35, 128) $\to$ Tanh $\to$ Linear(128, 128) $\to$ Tanh, producing hidden representation $h_{128} \in \mathbb{R}^{B \times 128}$.
- **Hydrological parameter head**: Linear(128, $N_{\text{mopex}}$) with physical parameter descaling.
- **Routing head**: Linear(128, 1) mapping to Unit Hydrograph parameter $\gamma_{\text{uh}}$.

### 2.3 Dedicated Hybrid Structure Encoder (`LearnedStructureNetHybridEncoder`)
- **Motivation**: Resolves *Shared-Backbone Representation Entanglement* (R16/R18), where the shared $h_{128}$ representation is dominated by bulk runoff flow tasks and lacks interception-aligned sub-spaces.
- **Input**: Concatenation of raw normalized basin attributes and detached hydrological representation:
  $$\mathbf{z}_{\text{struct}} = [\mathbf{x}_{35}, \text{stop\_gradient}(\mathbf{h}_{128})] \in \mathbb{R}^{B \times 163}$$
- **Encoder layers**:
  $$\mathbf{z}_1 = \text{Tanh}(\text{Linear}(163, 128)(\mathbf{z}_{\text{struct}}))$$
  $$\mathbf{z}_2 = \text{Tanh}(\text{Linear}(128, 64)(\mathbf{z}_1))$$
  $$\mathbf{logits}_{\text{struct}} = \text{Linear}(64, 8)(\mathbf{z}_2) \in \mathbb{R}^{B \times 4 \times 2}$$
- **Zero Gradient Leakage**: The $\text{stop\_gradient}$ on $h_{128}$ guarantees that structural loss $L_{\text{CF}}$ propagates strictly into the structure encoder (6 parameter tensors) and transmits zero gradient to the hydrological backbone.

---

## 3. Structural Supervision: Counterfactual Target Generation

### 3.1 Counterfactual Evidence $\Delta J$
For each basin $b$ and process $p \in \{0, 1, 2, 3\}$, the training loop generates online empirical counterfactual loss differences by evaluating the forward model with process $p$ forced ON ($w_p = 1.0$) versus forced OFF ($w_p = 0.0$):
$$\Delta J_{b,p} = J_{\text{OFF}}(b, p) - J_{\text{ON}}(b, p)$$
where $J(b, p) = L_{\text{fit}}(b, p) + \text{AIC}_{\text{pen}}(b, p)$.

The AIC penalty uses the canonical repository formulation:
$$\text{AIC}_{\text{pen}}(b, p) = \alpha_{\text{aic}} \cdot c_p \cdot \frac{N_{\text{total}}}{B \cdot \max(N_{v,b}, 1)}$$
with $\alpha_{\text{aic}} = 0.01$.

### 3.2 Soft-Target Probability $q$
$$\mathbf{q}_{b,p} = \sigma\left(\frac{\Delta J_{b,p}}{T_p}\right) \in (0, 1)$$
where temperature $T_p = \text{median}(\{|\Delta J_{b,p}| : b = 1 \dots B\})$ adaptively normalizes the dynamic scale of each process without arbitrary tuning.

### 3.3 Confidence-Weighted BCE Loss
To prevent uninformative near-zero counterfactual signal basins (the 73.6% ambiguous middle) from damping the gradient, we apply bounded confidence weighting (R17-B):
$$c_{b,p} = 2 \cdot |q_{b,p} - 0.5| \in [0, 1]$$
Normalized per process:
$$\tilde{c}_{b,p} = \frac{c_{b,p}}{\frac{1}{B}\sum_{i=1}^B c_{i,p} + 10^{-6}}$$
The structural loss is:
$$L_{\text{CF}} = \frac{1}{4 B} \sum_{p=0}^3 \sum_{b=1}^B \tilde{c}_{b,p} \cdot \text{BCE}(\sigma(\text{logit}_{b,p,1} - \text{logit}_{b,p,0}), q_{b,p})$$

---

## 4. Optimization & Gradient Isolation

### 4.1 Loss Composition
$$\mathcal{L}_{\text{total}} = L_{\text{fit\_aic}}(Q_{\text{sim}}(\mathbf{w}_{\text{active}}), Q_{\text{obs}}) + L_{\text{CF}}(\mathbf{logits}_{\text{struct}}, \mathbf{q})$$

### 4.2 Gradient Routing Matrix
| Parameter Subset | Gradient from $L_{\text{fit\_aic}}$ | Gradient from $L_{\text{CF}}$ |
|:---|:---:|:---:|
| Shared Backbone (`backbone`) | **YES** | **BLOCKED** ($\text{stop\_gradient}$) |
| Parameter Head (`params_head`) | **YES** | **NO** |
| Routing Head (`gamma_head`) | **YES** | **NO** |
| Structure Encoder (`structure_encoder`) | **BLOCKED** (masked/detached) | **YES** |

### 4.3 Unified Adadelta Optimizer
- **Optimizer**: Single canonical `torch.optim.Adadelta`
- **Parameters**: All 16 trainable parameter tensors (10 hydrological + 6 structure encoder)
- **Learning rate**: $\text{lr} = 1.0$ (no per-group lr tuning)
- **Gradient clipping**: $\text{clip\_grad\_norm\_}(\text{parameters}, 1.0)$

---

## 5. Canonical Replication Evidence (R19 Three-Seed Benchmark)

Evaluated on the canonical 5114-day out-of-sample window across 671 CAMELS basins at Epoch 10:

### 5.1 Streamflow Predictive Accuracy
| Metric | Seed 42 | Seed 43 | Seed 44 | 3-Seed Aggregate | Baseline Reference |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Median NSE** | **0.6518** | **0.6493** | **0.6494** | **0.6502 ± 0.0012** | 0.6317 (+0.0185) |
| **Mean NSE** | 0.5761 | 0.5744 | 0.5706 | 0.5737 ± 0.0023 | 0.5544 (+0.0193) |
| **Fraction NSE > 0** | 97.0% | 96.9% | 97.0% | 97.0% | 95.8% |
| **Fraction NSE > 0.5** | 74.5% | 74.4% | 74.4% | 74.4% | 71.2% |

### 5.2 Four-Process Structural Gate Separation ($\Delta = \bar{w}_{\text{pos}} - \bar{w}_{\text{zero}}$)
| Process Gate | $\Delta$ (Seed 42) | $\Delta$ (Seed 43) | $\Delta$ (Seed 44) | Mean $\Delta$ | Mean Spearman $\rho$ |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Vegetation Phenology (`w_phen`)** | +0.3009 | +0.2839 | +0.2831 | **+0.2893 ± 0.0082** | **+0.5800 ± 0.0233** |
| **Canopy Interception (`w_int`)** | **+0.1381** | **+0.1357** | **+0.1295** | **+0.1345 ± 0.0036** | **+0.3277 ± 0.0165** |
| **Snow Accum / Melt (`w_snow`)** | +0.4929 | +0.4633 | +0.4600 | **+0.4721 ± 0.0148** | **+0.7965 ± 0.0088** |
| **Subsurface Baseflow (`w_sub`)** | +0.2399 | +0.2041 | +0.2171 | **+0.2204 ± 0.0148** | **+0.4167 ± 0.0196** |

*All four processes demonstrate 100% sign-consistent positive separation and rank correlation across all seeds.*

---

## 6. How to Reproduce

### 6.1 Canonical Training Entry Point
```bash
cd project/flexmopex
python run_model.py \
  --config conf/config_flexmopex_canonical.yaml \
  --output-root results/canonical_freeze \
  --run-name seed_42
```

### 6.2 Canonical Evaluation
```bash
python scripts/evaluate_r19_adadelta_seed.py 42
python scripts/synthesize_r19_results.py
```

---

## 7. Known Limitations & Interpretation Guidelines

1. **Interception Separation Magnitude**: Interception ($\Delta \approx +0.135, \rho \approx +0.328$) is robust, reproducible, and non-collapsed, but quantitatively smaller than snow ($\Delta \approx +0.472, \rho \approx +0.797$). This reflects real hydrological physics: snow processes have strong, binary temperature signatures, whereas interception canopy storage produces subtle, distributed streamflow modifications.
2. **Oracle Ground Truth**: Oracle targets used in evaluation are computed purely as offline post-hoc diagnostic validation. Zero Oracle signal enters model training at any point.
