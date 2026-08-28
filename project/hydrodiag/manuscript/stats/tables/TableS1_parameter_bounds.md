# Table S1: Model Parameter Definitions and Optimization Bounds

| Parameter | Hydrological role | Unit | Lower bound | Upper bound | Applies to |
| :---: | :--- | :---: | :---: | :---: | :--- |
| k | Ratio of potential ET to reference crop evaporation | - | 0.5 | 2.0 | Shared host |
| b | Exponent of tension water storage capacity distribution curve | - | 0.1 | 2.0 | Shared host |
| i_m | Fraction of impervious and saturated direct runoff area | - | 0.0 | 0.3 | Shared host |
| u_m | Upper-layer soil tension water capacity | mm | 5.0 | 50.0 | Shared host |
| l_m | Lower-layer soil tension water capacity | mm | 20.0 | 200.0 | Shared host |
| d_m | Deep-layer soil tension water capacity | mm | 20.0 | 200.0 | Shared host |
| c | Deep-layer evapotranspiration coefficient | - | 0.05 | 0.3 | Shared host |
| s_m | Areal mean free water capacity of surface/shallow layer | mm | 5.0 | 100.0 | Shared host |
| ex | Exponent of free water capacity distribution curve | - | 0.1 | 2.0 | Shared host |
| k_i | Outflow coefficient from free water storage to interflow | d⁻¹ | 0.0 | 0.7 | Shared host |
| k_g | Outflow coefficient from free water storage to groundwater | d⁻¹ | 0.0 | 0.7 | Shared host |
| c_i | Recession constant of the linear interflow reservoir | - | 0.1 | 1.0 | Shared host |
| c_g | Recession constant of the linear groundwater reservoir | - | 0.9 | 1.0 | Shared host |
| a | Shape parameter of the Gamma unit hydrograph (Gamma-UH) | - | 0.0 | 2.9 | Shared host |
| θ | Scale parameter of the Gamma unit hydrograph (Gamma-UH) | d | 0.0 | 6.5 | Shared host |
| τ_warm | Warm-condition linear reservoir residence time / baseline smoothing | d | 0.0001 | 3.0 | TGD only |
| Δτ_cold | Additional cold-condition linear reservoir residence time increment | d | 0.1 | 180.0 | TGD only |
| C_TG | Snowpack thermal inertia and temperature weighting coefficient | - | 0.0 | 1.0 | CN only |
| K_f | Degree-day snowmelt factor ($D_f$) | mm °C⁻¹ d⁻¹ | 0.0 | 10.0 | CN only |

*Note*: Parameters 1–15 constitute the 15-dimensional core parameter space shared identically across Base, TGD, and CN configurations within the Xinanjiang (XAJ) host framework. TGD augments the host model with two generic temperature-dependent delay parameters ($\tau_{\mathrm{warm}}, \Delta\tau_{\mathrm{cold}}$). CN augments the host model with two degree-day snowpack accumulation and melt parameters ($C_{\mathrm{TG}}, K_f$). Boundaries define the feasible physical search space for both independent calibration (IC-CMA-ES) and differentiable parameter learning (dPL-MLP).
