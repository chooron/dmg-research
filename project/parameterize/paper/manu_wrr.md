# Evaluating the Reliability of Learned Basin Controls on Conceptual Hydrologic Model Parameters

Xin Jing^1^, Jungang Luo^1^, and Xue Yang^1^

^1^Department of Water Resources and Hydrology, Affiliation to be completed.

Corresponding author: Jungang Luo (jgluo@xaut.edu.cn)

## Keywords

Differentiable hydrologic modeling; Conceptual model parameters; HBV model; Parameter regionalization; Relationship reliability; Basin attributes; CAMELS-US; Distributional parameter learning; Hydrologic interpretability; Uncertainty diagnostics

## Highlights

- A relationship-reliability framework is proposed for interpreting learned HBV parameters.

- Predictive skill alone did not distinguish formulations for hydrologic interpretation.

- Distributional learning produced more reproducible attribute–parameter relationships.

- Seven HBV parameters formed a shared dominant-control core across formulations.

- Parameter uncertainty fields were structured but required coupling and boundary diagnostics.

## Abstract

Differentiable hydrologic models can learn conceptual model parameters from basin attributes, but adequate streamflow prediction does not ensure that the learned parameter field is reliable for hydrologic interpretation. We evaluate deterministic, Monte Carlo dropout, and distributional formulations for learning basin-specific static HBV parameters from CAMELS-US attributes across 531 basins, three loss functions, and five random seeds. We introduce a relationship-reliability framework that evaluates predictive adequacy, cross-seed and cross-loss stability of attribute–parameter relationships, cross-formulation dominant-control consistency, and environmental-gradient coherence. All three formulations produced adequate simulations, and predictive skill alone did not differentiate them for interpretation. Relationship-level diagnostics indicated higher reproducibility for the distributional formulation, with lower cross-seed and cross-loss variability in Spearman attribute–parameter correlations and a more internally consistent relationship matrix. Across formulations, seven of fourteen parameters formed a shared dominant-control core, including links between slope and runoff nonlinearity, potential evapotranspiration and storage capacity, aridity and percolation, and soil conductivity and upper-zone behavior. Distributional means expressed these relationships as environmental gradients, while uncertainty fields provided structured but bounded diagnostic information. These results suggest that relationship reliability can serve as a screening criterion for interpreting learned behavioral parameter gradients.

# 1 Introduction

Hydrologic models are expected to support both prediction and interpretation. In regional applications, this expectation is particularly demanding because model parameters must often be transferred across basins rather than calibrated independently at every gauge. Regionalization methods therefore rely on the assumption that basin descriptors, including climate, topography, vegetation, soil, and geology, contain information about hydrologic model parameters (Samaniego et al., 2010). Large-sample data sets such as CAMELS-US have made this assumption testable across diverse hydroclimatic and physiographic settings (Addor et al., 2017; Newman et al., 2015). The central question is no longer only whether regionalized models can predict streamflow, but whether the learned parameter fields can support hydrologic reasoning about basin controls.

Purely data-driven deep learning models have achieved strong streamflow prediction in large-sample hydrology, but their internal representations are difficult to interpret as hydrologic states, fluxes, or parameters. Differentiable hydrologic modeling provides a different path. A process-based or conceptual model is placed inside a differentiable computational graph, and neural networks are used to provide learnable components, such as parameterization functions or selected model modules. This allows parameters to be learned directly from streamflow data while retaining a model structure that represents storage, runoff generation, evapotranspiration, snow processes, groundwater exchange, or routing. Recent differentiable, learnable process-based models have shown that such hybrid structures can approach strong predictive performance while retaining process-related outputs and interpretable model components (Feng et al., 2022; Feng et al., 2023).

These developments have opened an important opportunity for regional hydrology, but they also expose a specific interpretive gap. Existing differentiable parameter-learning studies have mainly evaluated predictive skill, spatial generalization, untrained physical variables, or the coherence of inferred parameter and flux fields. These analyses establish the promise of differentiable learning as a regionalized modeling framework. However, they do not directly test whether the learned relationships between basin attributes and conceptual model parameters are reproducible across plausible training perturbations. A spatially coherent parameter field, an improved hydrograph, or an interpretable flux diagnostic does not by itself show that individual learned parameters have stable and hydrologically meaningful relationships with basin properties.

This distinction is important because conceptual hydrologic parameters play two roles at once. They help reproduce discharge, and they provide a compact language for discussing storage capacity, runoff nonlinearity, snow accumulation and melt, recession behavior, and routing. In conventional calibration, hydrologists are already cautious about equifinality, process compensation, parameter bounds, data limitations, and objective-function dependence (Beven & Freer, 2001). Differentiable learning changes the calibration mechanism but does not remove these concerns. It adds a neural mapping from basin attributes to parameters, and this mapping can use attributes successfully for prediction while still producing parameter-attribute relationships that depend on random initialization, loss function, or learning formulation. For hydrologic interpretation, the relevant question is therefore whether learned basin controls on parameters remain reproducible when the training procedure is reasonably perturbed.

We address this question through relationship reliability. In this study, relationship reliability refers to the reproducibility of learned basin attribute-parameter relationships across random seeds, loss functions, and plausible learning formulations. We use it as a prerequisite for interpreting learned conceptual parameters rather than as a post hoc supplement to streamflow metrics. This framing organizes the analysis into four linked checks: predictive adequacy, relationship stability across seeds and losses, dominant-control consistency across formulations, and environmental-gradient coherence in the resulting parameter fields. Together, these checks form an evidence hierarchy in which hydrologic interpretation is considered only after the learned relationships have shown sufficient reproducibility.

The application is differentiable HBV parameter learning over CAMELS-US basins. We compare three formulations that share the same differentiable HBV backbone but differ in how the attribute-to-parameter mapping is represented. The deterministic formulation maps basin attributes to one point-estimate parameter vector. The Monte Carlo dropout formulation provides a stochastic neural baseline by retaining dropout at inference. The distributional formulation predicts an attribute-conditioned parameter distribution and therefore produces both parameter means and parameter-scale spread. It is included not to approximate a full Bayesian posterior, but to test whether explicit parameter-scale uncertainty learning affects the reproducibility of learned basin controls and whether the resulting uncertainty fields carry structured diagnostic information.

The model comparison is therefore used to classify evidence rather than to select a winner from predictive skill alone. The deterministic and dropout formulations provide alternative ways in which a neural HBV parameter field can be learned, and they help identify relationships that are common across plausible formulations. A relationship that remains stable across seeds and losses and is shared across formulations has a different evidentiary status from one that appears only under one learning configuration. Similarly, a distributional uncertainty gradient is informative only after checking whether it reflects structured basin information rather than mean coupling or simple boundary effects. This comparison shifts the manuscript from a general model-ranking exercise to a hydrologic question: when are learned conceptual parameters reliable enough to interpret?

Our contribution follows this evidence hierarchy. First, we evaluate predictive adequacy to ensure that learned parameter fields are assessed under credible streamflow simulations. Second, we formalize relationship-stability metrics for learned conceptual parameters, including cross-seed variability, cross-loss variability, sign consistency, top-k overlap, and matrix-level compactness. Third, we evaluate dominant-control consistency across deterministic, dropout, and distributional formulations to distinguish shared basin controls from formulation-dependent controls. Fourth, we examine environmental-gradient coherence in distributional parameter means and assess whether distributional parameter uncertainty contains structured diagnostic information after screening for mean-standard deviation coupling and boundary effects.

The interpretation remains intentionally bounded. HBV parameters are behavioral summaries of a conceptual model, not direct measurements of individual physical properties, and attribute collinearity means that a stable relationship may represent a broader environmental gradient rather than an isolated causal control. Within this scope, reproducible relationships can still provide useful behavioral signatures for regional hydrology. The remainder of the paper follows the proposed evidence hierarchy. The Methods define the learned mappings, normalized parameters, reliability metrics, and uncertainty diagnostics. The Results first establish predictive adequacy, then test relationship reliability, identify shared basin controls, and examine environmental gradients in distributional means and uncertainties. The Discussion returns to the hydrologic meaning of reproducible behavioral gradients and the limits imposed by compensation, collinearity, and bounded parameter transformations.

# 2 Materials and Methods

## 2.1 Differentiable HBV parameter learning

Differentiable parameter learning links regionalized parameter estimation with process-based hydrologic simulation. Instead of calibrating a separate parameter vector for each basin, a neural network learns an attribute-to-parameter mapping from a large sample of basins. For basin \(b\), let \(\mathbf{x}_b\) denote the static basin-attribute vector, and let \(m\), \(l\), and \(s\) index the parameter-estimation formulation, loss function, and random seed. The general learned mapping is

$$
\boldsymbol{\theta}_{b}^{(m,l,s)}
=
f_m\left(\mathbf{x}_b;\boldsymbol{\omega}^{(m,l,s)}\right),
\tag{1}
$$

where \(\boldsymbol{\theta}_{b}^{(m,l,s)}\) is the learned parameter vector and \(\boldsymbol{\omega}^{(m,l,s)}\) denotes the trained neural-network weights. The hydrologic simulation driven by \(\boldsymbol{\theta}_{b}^{(m,l,s)}\) remains inside the computational graph, so the neural mapping can be trained directly against streamflow observations through backpropagation.

The hydrologic backbone is an HBV-type conceptual model implemented in a differentiable framework. HBV represents snow accumulation and melt, soil water storage, evapotranspiration limitation, percolation, upper and lower groundwater storage, and runoff generation through a compact set of process-related parameters (Bergstrom, 1995). Routing is represented using a gamma unit hydrograph. The analyzed parameter set therefore includes both HBV process parameters and routing parameters: \(\mathrm{BETA}\), \(\mathrm{FC}\), \(\mathrm{LP}\), \(\mathrm{PERC}\), \(\mathrm{UZL}\), \(\mathrm{K}_0\), \(\mathrm{K}_1\), \(\mathrm{K}_2\), \(\mathrm{TT}\), \(\mathrm{CFMAX}\), \(\mathrm{CFR}\), \(\mathrm{CWH}\), \(\mathrm{UH}_a\), and \(\mathrm{UH}_b\). Detailed HBV equations, parameter ranges, and input forcing definitions are provided in Supporting Information Text S1 and Tables S1-S2.

For analyses that compare parameters with different units and search ranges, parameter values are normalized as

$$
\tilde{\theta}_{b,j}
=
\frac{\theta_{b,j}-L_j}{U_j-L_j},
\tag{2}
$$

where \(\theta_{b,j}\) is the physical value of parameter \(j\), and \(L_j\) and \(U_j\) are its lower and upper bounds. The learned parameters are interpreted as behavioral parameters of the adopted differentiable HBV model, not as direct observations of true physical properties.

## 2.2 Parameter-estimation formulations

We compared three formulations for estimating static HBV parameters from basin attributes under the same differentiable HBV backbone. The formulations differ in how they represent the attribute-conditioned parameter field and how parameter spread is obtained.

The deterministic formulation provides a point-estimate baseline. A multilayer perceptron maps the static attribute vector \(\mathbf{x}_b\) to one normalized parameter vector,

$$
\tilde{\boldsymbol{\theta}}_b
=
f_{\mathrm{det}}\left(\mathbf{x}_b;\boldsymbol{\omega}_{\mathrm{det}}\right).
\tag{3}
$$

This formulation produces one parameter vector per basin for each trained model, loss function, and random seed. It therefore represents the simplest differentiable regionalization setting, in which parameter spread is not explicitly represented.

The Monte Carlo dropout formulation uses the same attribute-to-parameter learning principle but retains dropout during inference. Repeated stochastic forward passes produce parameter samples,

$$
\tilde{\boldsymbol{\theta}}_{b}^{(r)}
=
f_{\mathrm{mcd}}\left(\mathbf{x}_b;\boldsymbol{\omega}_{\mathrm{mcd}},\mathbf{d}^{(r)}\right),
\qquad r=1,\ldots,R_{\mathrm{mcd}},
\tag{4}
$$

where \(\mathbf{d}^{(r)}\) is the dropout mask for the \(r\)-th forward pass. In this study, \(R_{\mathrm{mcd}}=100\) stochastic passes were used. The sample mean was used as the basin-level parameter estimate, and the sample standard deviation was used as a measure of dropout-induced spread. This spread reflects stochasticity in the neural parameterization rather than an explicitly learned parameter-scale distribution.

The distributional formulation predicts a latent location and log-standard deviation for each basin and parameter,

$$
\boldsymbol{\mu}_b,\log\boldsymbol{\sigma}_b
=
g_{\mathrm{dist}}\left(\mathbf{x}_b;\boldsymbol{\omega}_{\mathrm{dist}}\right),
\tag{5}
$$

with \(\log\boldsymbol{\sigma}_b\) constrained to \([-5,2]\) during training. Latent samples are generated using the reparameterization form

$$
\mathbf{z}_{b}^{(r)}
=
\boldsymbol{\mu}_b
+
\boldsymbol{\sigma}_b\odot\boldsymbol{\epsilon}^{(r)},
\qquad
\boldsymbol{\epsilon}^{(r)}\sim\mathcal{N}(\mathbf{0},\mathbf{I}),
\tag{6}
$$

and transformed to the bounded parameter scale by \(T(\cdot)\). During training, one latent parameter sample is drawn for each forward simulation. The distributional loss combines the hydrologic loss with KL regularization,

$$
\mathcal{L}_{\mathrm{dist}}
=
\mathcal{L}_{\mathrm{hydro}}
+
\beta_{\mathrm{KL}}
D_{\mathrm{KL}}
\left[
\mathcal{N}\left(\boldsymbol{\mu}_b,\mathrm{diag}(\boldsymbol{\sigma}_b^2)\right)
\Vert
\mathcal{N}(\mathbf{0},\mathbf{I})
\right].
\tag{7}
$$

The KL coefficient was \(10^{-3}\) and was linearly warmed up over the first 10 epochs. For inference and relationship analysis, \(R_{\mathrm{dist}}=100\) bounded parameter samples were used to estimate the parameter mean and standard deviation. The distributional standard deviation is used as a structured diagnostic quantity, not as a complete Bayesian posterior or as direct evidence of physical parameter identifiability.

## 2.3 Data, inputs, and experimental configuration

CAMELS-US was used as the large-sample test bed because it provides daily meteorological forcing, streamflow observations, static basin attributes, and basin locations for catchments across the contiguous United States. The analysis used a 531-basin subset with complete learned-parameter outputs, matched attributes, and matched coordinates. Each complete parameter run contained 7,434 basin-parameter entries, corresponding to 531 basins and 14 HBV and routing parameters.

The neural parameterization network receives static CAMELS basin attributes only, including hydroclimatic, topographic, vegetation, soil, and geological descriptors. These descriptors are basin-scale quantities, so all attribute-parameter analyses are cross-basin analyses rather than temporal analyses. The differentiable HBV simulation is driven separately by daily precipitation, mean air temperature, and potential evapotranspiration. Streamflow observations are converted from volumetric discharge to basin-area-normalized depth units before model training and metric calculation. The full attribute list and input preprocessing are provided in Supporting Information Text S2 and Table S2.

Training used 1 January 1989 to 31 December 1998, and testing used 1 January 1999 to 31 December 2009. A 365-day warm-up period was removed before comparing simulated and observed streamflow. No separate validation period was used; all runs were trained for 100 epochs under the same training protocol and evaluated on the independent test period.

The experimental matrix crossed three parameter-estimation formulations, three loss functions, and five random seeds. The formulations were deterministic point estimation, Monte Carlo dropout, and distributional parameter learning. The loss functions were `HybridNseBatchLoss`, `NseBatchLoss`, and `LogNseBatchLoss`. The random seeds were 111, 222, 333, 444, and 555, yielding 45 complete runs. Loss functions were treated as structured perturbations because they emphasize different parts of the hydrograph and may induce different parameter-compensation pathways.

All formulations used a static attribute-to-parameter network with two hidden ReLU layers and hidden size 128. The MC-dropout formulation inserted dropout after each hidden layer with dropout rate 0.2. Training used Adam with learning rate \(10^{-3}\), a step learning-rate scheduler with step size 20 and decay factor 0.5, and a training batch size of 100 basins. Additional implementation settings are provided in Supporting Information Text S3.

Run-level processing was performed before analysis. Parameter tables were filtered to complete 531-basin runs and the 14 analyzed parameters. Duplicate logical rows were diagnosed and collapsed by model formulation, loss function, seed, basin, and parameter. All downstream analyses used the deduplicated run inventory. Deterministic outputs were interpreted as point estimates, whereas MC-dropout and distributional outputs provided sample means and sample standard deviations on the parameter scale.

## 2.4 Evaluation metrics and diagnostic analyses

### 2.4.1 Predictive adequacy and parameter-value diagnostics

Streamflow metrics were used to establish predictive adequacy, not to select the interpretive formulation. Nash-Sutcliffe efficiency, Kling-Gupta efficiency, absolute bias, and absolute percent bias were computed basin by basin on the test period after warm-up removal and then summarized across basins and runs. Metric formulas and implementation details are provided in Supporting Information Text S4.

Parameter-value stability was evaluated across random seeds within each model formulation, loss function, basin, and parameter using the standard deviation and range of normalized parameter values. These diagnostics describe numerical parameter stability but are not interpreted as relationship reliability. Boundary diagnostics were included because low apparent variability can arise from saturation at parameter bounds. Boundary saturation was computed using a primary near-boundary threshold of \(\tilde{\theta}\leq0.02\) or \(\tilde{\theta}\geq0.98\), with a secondary 0.05/0.95 threshold used for sensitivity checks. Distance to the nearest boundary was defined as

$$
d_{b,j}
=
\min
\left(
\tilde{\theta}_{b,j},
1-\tilde{\theta}_{b,j}
\right).
\tag{8}
$$

For MC-dropout and distributional formulations, stochastic interval diagnostics were computed from 100 bounded parameter samples per run, with the 90% interval width \(q_{0.95}-q_{0.05}\) used as the main spread diagnostic.

### 2.4.2 Relationship reliability metrics

Relationship reliability was evaluated at three levels. Pair-level metrics evaluate the stability of individual attribute-parameter relationships. Parameter-level metrics evaluate whether the dominant basin controls for each HBV parameter are reproducible. Matrix-level metrics evaluate the consistency of the full attribute-by-parameter relationship structure.

For model formulation \(m\), loss function \(l\), seed \(s\), basin attribute \(a\), and HBV parameter \(j\), the primary relationship statistic was the basin-level Spearman correlation

$$
\rho_{a,j}^{(m,l,s)}
=
\mathrm{corr}_S
\left(
x_{b,a},
\tilde{\theta}_{b,j}^{(m,l,s)}
\right)_{b=1}^{B},
\qquad B=531.
\tag{9}
$$

Spearman correlation was used because many attribute-parameter relationships are expected to be monotonic but not necessarily linear. Cross-seed relationship variability was computed within each model and loss as

$$
\mathrm{SD}_{seed}
\left(\rho_{a,j}^{(m,l)}\right)
=
\sqrt{
\frac{1}{S-1}
\sum_{s=1}^{S}
\left(
\rho_{a,j}^{(m,l,s)}
-
\bar{\rho}_{a,j}^{(m,l)}
\right)^2
},
\tag{10}
$$

where \(S=5\). Cross-loss variability was computed after seed aggregation,

$$
\mathrm{SD}_{loss}
\left(\rho_{a,j}^{(m)}\right)
=
\sqrt{
\frac{1}{L-1}
\sum_{l=1}^{L}
\left(
\bar{\rho}_{a,j}^{(m,l)}
-
\bar{\rho}_{a,j}^{(m)}
\right)^2
},
\tag{11}
$$

where \(L=3\). Seed and loss ranges were also reported as complementary sensitivity summaries.

Sign consistency was recorded as the fraction of nonzero correlations sharing the majority sign. Candidate high-magnitude relationships were selected using a top-\(k\) rule on absolute Spearman correlation. The top-10 rule within each parameter was used for broad relationship-stability summaries, whereas the top-3 rule was used for focused dominant-control classification. The selection was applied across all formulations, losses, and seeds to avoid selecting relationships that were favorable to only one formulation.

Dominant controls were classified by comparing modal dominant attributes and directions across deterministic, MC-dropout, and distributional formulations. A control was classified as shared when all three formulations identified the same dominant attribute with consistent direction, partially shared when two formulations agreed or when all formulations indicated closely related dominant controls, and model-sensitive when dominant attributes or directions differed substantially. Matrix-level reliability was evaluated by representing each run as a full attribute-by-parameter Spearman correlation matrix \(\mathbf{R}\). Pairwise matrix separation was summarized using the Frobenius distance,

$$
D_F(\mathbf{R}_p,\mathbf{R}_q)
=
\left\|
\mathbf{R}_p-\mathbf{R}_q
\right\|_F.
\tag{12}
$$

Cosine similarity and matrix correlation were also computed from vectorized finite matrix entries. Within-formulation compactness was defined as the mean within-formulation pairwise matrix correlation across runs, with larger values indicating a more internally consistent relationship structure.

### 2.4.3 Environmental gradients and uncertainty diagnostics

Environmental-gradient diagnostics were applied after relationship-reliability metrics were computed, with detailed mean and uncertainty analyses performed for the formulation selected for interpretation in the Results. These diagnostics test whether reliable attribute-parameter relationships also appear as coherent basin-scale gradients, without treating the gradients as direct causal effects.

Gradient analyses used two complementary summaries. First, basin-level Spearman correlations were computed between selected attributes and normalized parameter means or normalized parameter standard deviations. Second, basins were assigned to rank-based low, middle, and high terciles for each selected gradient. For each gradient-parameter pair, the high-minus-low tercile contrast was defined as

$$
\Delta_{H-L}(a,j)
=
\mathrm{median}
\left(
\tilde{\theta}_{b,j}: b\in H_a
\right)
-
\mathrm{median}
\left(
\tilde{\theta}_{b,j}: b\in L_a
\right),
\tag{13}
$$

where \(H_a\) and \(L_a\) are the high and low terciles of attribute \(a\). Group summaries report medians and interquartile ranges; high-low group tests were used as descriptive contrasts rather than causal tests.

Parameter uncertainty was analyzed on the normalized parameter scale. For basin \(b\) and parameter \(j\), normalized parameter uncertainty was defined as

$$
u_{b,j}
=
\frac{s_{\theta,b,j}}
{U_j-L_j},
\tag{14}
$$

where \(s_{\theta,b,j}\) is the stochastic parameter standard deviation in physical parameter units. Mean-standard deviation coupling was measured as the Spearman correlation between normalized parameter mean and normalized parameter standard deviation for each parameter. Boundary sensitivity was measured using the Spearman correlation between distance-to-boundary and parameter standard deviation, together with the share of basins near the boundary. Uncertainty-attribute relationships were flagged as mean-coupled when the absolute mean-standard deviation correlation was at least 0.5 and as boundary-sensitive when the absolute boundary-distance-standard deviation correlation was at least 0.4 or the near-boundary share was at least 0.25. These flags constrain uncertainty claims to diagnostic structure rather than physical identifiability.

# 3 Results

### 3.1 Predictive performance and parameter-value diagnostics

All three formulations produced adequate streamflow simulations across the 531 CAMELS-US basins (Figure 1 and Table 1). Across complete model runs, median NSE was 0.611 for \(\delta_{\mathrm{base}}\), 0.624 for \(\delta_{\mathrm{mcd}}\), and 0.611 for \(\delta_{\mathrm{dist}}\); median KGE was 0.626, 0.639, and 0.625, respectively. Under the reference hybrid loss, median NSE ranged from 0.632 to 0.636 and median KGE from 0.675 to 0.679. The empirical distributions also overlapped closely across the three formulations. The fractions of basins exceeding NSE = 0.5 were 0.74, 0.74, and 0.73 for \(\delta_{\mathrm{base}}\), \(\delta_{\mathrm{mcd}}\), and \(\delta_{\mathrm{dist}}\), respectively. The corresponding fractions exceeding KGE = 0.5 were 0.83, 0.81, and 0.83. Pairwise basin-level comparisons against \(\delta_{\mathrm{base}}\) also showed that most NSE and KGE values for \(\delta_{\mathrm{mcd}}\) and \(\delta_{\mathrm{dist}}\) were close to the one-to-one line. These predictive summaries indicate that the three learned parameter fields were evaluated under broadly comparable streamflow-simulation conditions.

The loss-specific summaries showed the same overall pattern (Figure S1). Under the NSE loss, median NSE values were 0.65, 0.66, and 0.65 for \(\delta_{\mathrm{base}}\), \(\delta_{\mathrm{mcd}}\), and \(\delta_{\mathrm{dist}}\), respectively; the corresponding median KGE values were 0.65, 0.66, and 0.65. Under the logNSE loss, median NSE values were lower for all formulations, with values of 0.56, 0.58, and 0.56, while median KGE values were 0.53, 0.58, and 0.53. Under the hybrid loss, median NSE values were 0.63, 0.64, and 0.63, and median KGE values were 0.68, 0.68, and 0.68. Differences among loss functions were comparable to, and in some cases larger than, differences among formulations. Bias metrics gave a similar formulation-level pattern. The dropout formulation had slightly higher overall median NSE and KGE, but it also had slightly larger median absolute bias than the deterministic and distributional formulations. Predictive metrics were therefore used as an adequacy screen, rather than as the main criterion for selecting an interpretive formulation.

Raw parameter-value diagnostics provided a second preliminary check before the relationship analyses (Figure 2). Apparent cross-seed stability varied by parameter and loss function, and several parameters with low apparent variability were close to the prescribed search-range boundaries. Boundary saturation was especially visible for snow, recession, and routing parameters. After boundary-sensitive parameters were separated from the pooled summaries, formulation differences in raw parameter variability became smaller and were not uniformly favorable to \(\delta_{\mathrm{dist}}\). A parameter-value versus relationship-stability diagnostic further identified cases in which low raw variability did not correspond to stable attribute–parameter relationships (Table S3). These cases included deterministic \(\mathrm{CFR}\), \(\mathrm{CWH}\), and \(\mathrm{UH}_a\), and dropout \(\mathrm{CFR}\), \(\mathrm{CWH}\), and \(\mathrm{K}_2\), several of which also had high boundary saturation. Thus, numerical parameter stability and attribute–parameter relationship stability were treated as separate diagnostics in the subsequent analyses.

### 3.2 Relationship-level stability across training perturbations and hydroclimatic groups

Relationship-level diagnostics provided a more differentiated comparison among the three formulations than predictive metrics or raw parameter-value summaries (Figure 3 and Table 2). Candidate relationships were selected by mean absolute Spearman correlation across all formulations, losses, and seeds, so the comparison was not defined around \(\delta_{\mathrm{dist}}\) alone. Within this common candidate set, the median cross-seed standard deviation of Spearman \(\rho\) was 0.0241 for \(\delta_{\mathrm{dist}}\), compared with 0.0347 for \(\delta_{\mathrm{mcd}}\) and 0.0461 for \(\delta_{\mathrm{base}}\). Median seed ranges followed the same ordering: 0.0630 for \(\delta_{\mathrm{dist}}\), 0.0891 for \(\delta_{\mathrm{mcd}}\), and 0.1122 for \(\delta_{\mathrm{base}}\).

The full relationship-sensitivity summary gave the same formulation ordering. Mean cross-seed SD of Spearman \(\rho\) was 0.0308 for \(\delta_{\mathrm{dist}}\), 0.0546 for \(\delta_{\mathrm{mcd}}\), and 0.0663 for \(\delta_{\mathrm{base}}\). The pooled distributions in Figure 3a show lower and more concentrated seed SD values for \(\delta_{\mathrm{dist}}\), whereas \(\delta_{\mathrm{base}}\) and \(\delta_{\mathrm{mcd}}\) had wider upper tails. Parameter-level summaries in Figure 3b show that lower seed sensitivity under \(\delta_{\mathrm{dist}}\) was visible for several runoff-generation and storage parameters, including \(\mathrm{BETA}\), \(\mathrm{FC}\), \(\mathrm{PERC}\), and \(\mathrm{UZL}\). Snow, recession, and routing parameters retained larger seed-sensitivity ranges, indicating that relationship stability remained parameter dependent.

Dominant-control diagnostics showed a similar pattern. Median dominant-attribute consistency across seeds was 1.000 for \(\delta_{\mathrm{dist}}\), compared with 0.800 for both \(\delta_{\mathrm{base}}\) and \(\delta_{\mathrm{mcd}}\). Median top-5 overlap was 0.700 for \(\delta_{\mathrm{dist}}\), 0.679 for \(\delta_{\mathrm{mcd}}\), and 0.629 for \(\delta_{\mathrm{base}}\) (Figure 3c). The representative relationship panels in Figure 3d–g show stable signs and comparable magnitudes across the five seeds for four high-magnitude relationships: \(\mathrm{slope\_mean}\)–\(\mathrm{BETA}\), \(\mathrm{pet\_mean}\)–\(\mathrm{FC}\), \(\mathrm{aridity}\)–\(\mathrm{PERC}\), and \(\mathrm{soil\_conductivity}\)–\(\mathrm{UZL}\). These panels provide examples of relationships included in the stability summary, rather than a complete display of all parameter–attribute pairs.

Loss-function sensitivity was larger than seed sensitivity, but the relative ordering among formulations was retained. Across loss functions, \(\delta_{\mathrm{dist}}\) had the lowest mean cross-loss SD of Spearman \(\rho\), with a value of 0.129, compared with 0.142 for \(\delta_{\mathrm{base}}\) and 0.168 for \(\delta_{\mathrm{mcd}}\) (Table 2). The top-\(k\) sensitivity analysis gave the same qualitative ordering across alternative relationship-selection thresholds (Figure S3). This analysis was used to check whether the relationship-stability result depended on the number of retained high-correlation pairs.

Hydroclimatic-group analyses showed that the relationship-stability pattern was not restricted to the full-sample aggregation (Figure 4). The 531 basins were separated into seven hydroclimatic strata, covering humid steep, low-snow humid lowland, arid lowland, arid seasonal, low-snow arid steep, snow arid steep, and snow humid steep settings (Figure 4a). Within these strata, \(\delta_{\mathrm{dist}}\) generally maintained lower seed-to-seed variability in high-correlation relationships (Figure 4b). Its median group-wise seed SD of Spearman \(\rho\) was 0.0491, compared with 0.0586 for \(\delta_{\mathrm{base}}\) and 0.0541 for \(\delta_{\mathrm{mcd}}\). Group-wise top-5 overlap was also comparable or higher for \(\delta_{\mathrm{dist}}\) in several strata (Figure 4c), although overlap varied among groups.

The hydroclimatic strata also showed organized differences in learned parameter fields (Figure 4d–e). Median normalized parameter values differed among strata for several snow, storage, recession, and routing parameters. The within-group IQR heatmaps showed that some parameters retained larger within-stratum dispersion than others. These parameter-value summaries were used as descriptive context for the relationship analysis; the formulation comparison in this section was based on relationship stability rather than on group-wise parameter values alone. A leave-one-hydroclimatic-group-out check for the main distributional relationships did not identify any case with \(|\Delta \rho| \ge 0.15\), indicating that the full-sample correlations were not controlled by one hydroclimatic stratum alone (Figure S14; Table S4). Within-group correlations were more variable than the leave-one-group-out summaries. For example, the \(\mathrm{FC}\)–PET relationship weakened in several groups and changed sign in the snow arid steep group, and the \(\mathrm{PERC}\)–aridity relationship changed sign in the low-snow arid steep group. These group-wise summaries were retained as sensitivity diagnostics and were not used as ungauged-basin validation. A deduplication sensitivity check produced negligible changes for \(\delta_{\mathrm{dist}}\): mean cross-seed SD changed from 0.03120 before deduplication to 0.03131 after deduplication, and the main shared-control correlations changed by less than 0.01.

### 3.3 Shared dominant basin controls across formulations

The three formulations recovered a common but nonuniform basin-control structure rather than unrelated attribute–parameter relationship matrices (Tables 3 and 4). Of the 14 parameters, 7 were classified as shared dominant controls, 6 as partially shared controls, and 1 as model-sensitive. The shared group comprised \(\mathrm{BETA}\), \(\mathrm{FC}\), \(\mathrm{K}_2\), \(\mathrm{PERC}\), \(\mathrm{TT}\), \(\mathrm{UZL}\), and \(\mathrm{UH}_a\). For these parameters, the dominant attribute and relationship direction were consistent across \(\delta_{\mathrm{base}}\), \(\delta_{\mathrm{mcd}}\), and \(\delta_{\mathrm{dist}}\). Figure 6 shows the corresponding distributional relationship matrix and group-level summaries.

Several shared controls had relatively high magnitude in the distributional relationship matrix. The main examples were the negative \(\mathrm{slope\_mean}\)–\(\mathrm{BETA}\) relationship, the positive \(\mathrm{pet\_mean}\)–\(\mathrm{FC}\) relationship, the negative \(\mathrm{aridity}\)–\(\mathrm{PERC}\) relationship, and the positive \(\mathrm{soil\_conductivity}\)–\(\mathrm{UZL}\) relationship (Figure 6a; Table 3). In the distributional formulation, these relationships had Spearman \(\rho\) values of -0.583, 0.509, -0.594, and 0.570, respectively. In Figure 6a, these pairs appear as dominant or stable high-magnitude entries in distinct descriptor groups, linking \(\mathrm{BETA}\) to terrain, \(\mathrm{FC}\) to evaporative demand, \(\mathrm{PERC}\) to hydroclimatic dryness, and \(\mathrm{UZL}\) to soil hydraulic properties.

The attribute-group summaries in Figure 6b–c show how relationship strength varied by parameter and descriptor group within the distributional formulation. Snow-related parameters were most strongly associated with climate descriptors and secondarily with topography; their top-3 mean absolute correlations were 0.70 for climate and 0.55 for topography. Soil-related parameters had top-3 mean absolute correlations of 0.47 with climate, 0.46 with soil, and 0.42 with topography. Production parameters showed broader multi-group associations, with top-3 mean absolute correlations of 0.54 with climate, 0.51 with soil, and 0.49 with vegetation descriptors. Routing parameters had more moderate group-level associations, ranging from 0.40 to 0.46 across climate, topography, vegetation, and soil, and a weaker association with geology (0.28). At the parameter level, \(\mathrm{CWH}\), \(\mathrm{PERC}\), and \(\mathrm{UZL}\) had among the larger top-\(k\) mean relationship magnitudes, whereas \(\mathrm{LP}\), \(\mathrm{K}_2\), and some routing-related entries were lower in the same summary.

The remaining parameters were less uniform across formulations (Tables 3 and 4). \(\mathrm{CFR}\), \(\mathrm{CWH}\), \(\mathrm{CFMAX}\), \(\mathrm{K}_0\), \(\mathrm{K}_1\), and \(\mathrm{UH}_b\) were classified as partially shared controls, reflecting shifts in dominant attribute, reduced agreement in rank, or weaker directional consistency across formulations. Among these, \(\mathrm{UH}_b\) was additionally flagged by sign inconsistency. \(\mathrm{LP}\) was the only parameter classified as model-sensitive. This classification separated a shared dominant-control group from parameters whose dominant relationships varied more across formulations.

Matrix-level diagnostics were consistent with the parameter-level classification (Table 4). The distributional formulation had the highest within-formulation compactness (0.809), compared with 0.702 for \(\delta_{\mathrm{base}}\) and 0.680 for \(\delta_{\mathrm{mcd}}\). Its mean within-formulation Frobenius distance was also lower, 2.624 versus 3.453 for \(\delta_{\mathrm{base}}\) and 3.668 for \(\delta_{\mathrm{mcd}}\). Intermodel similarity remained substantial, so \(\delta_{\mathrm{dist}}\) occupied the same broad relationship structure shared across formulations. The spatial maps of distributional parameter means provide descriptive spatial context for these learned parameter fields (Figure 5). Several maps showed regional organization in snow, storage, recession, and routing parameters, including contrasts between western mountainous regions, arid basins, and humid eastern basins.

### 3.4 Environmental gradients of distributional parameter means and uncertainties

Distributional parameter means expressed several dominant controls as environmental gradients (Figures 8 and 9; Table 5). In the tercile summaries, \(\mathrm{BETA}\) decreased with slope, \(\mathrm{FC}\) increased with PET, \(\mathrm{PERC}\) decreased with aridity, and \(\mathrm{UZL}\) increased with soil conductivity (Figure 8). The corresponding high-minus-low median differences were -0.40 for \(\mathrm{BETA}\) across slope terciles, +0.20 for \(\mathrm{FC}\) across PET terciles, -0.23 for \(\mathrm{PERC}\) across aridity terciles, and +0.26 for \(\mathrm{UZL}\) across soil-conductivity terciles. These four gradients match the dominant distributional relationships shown in Figure 6 and reported in Table 5.

Snow-related parameter means also showed large environmental gradients in the distributional formulation. \(\mathrm{CWH}\) decreased with snow fraction, with Spearman \(\rho=-0.90\) and a high-minus-low median difference of -0.34. \(\mathrm{CFR}\) also decreased with snow fraction, with \(\rho=-0.62\) and a high-minus-low difference of -0.08 (Figure 8). These snow-related relationships were assigned a partially shared, rather than fully shared, cross-formulation evidence class in Table 3. The snow-related tercile panels therefore provide distributional-gradient evidence, while the cross-formulation classification remains more conservative.

The detailed gradient panels in Figure 9 show that several relationships were monotonic but not strictly linear. The \(\mathrm{CWH}\)-snow fraction relationship declined sharply at low snow fraction and then flattened. The \(\mathrm{BETA}\)-slope relationship decreased most strongly from low to moderate slopes, while the \(\mathrm{UZL}\)-soil conductivity relationship increased across the lower and middle parts of the soil-conductivity range. \(\mathrm{FC}\) increased with PET, and \(\mathrm{PERC}\) decreased with aridity but showed local variation in the binned median curve. Additional panels in Figure 9 showed secondary gradients for \(\mathrm{K}_1\), \(\mathrm{K}_2\), \(\mathrm{UH}_b\), and \(\mathrm{TT}\). These panels provide detailed visual support for the rank-based and tercile summaries, including relationships that were weaker or less consistently classified across formulations.

Distributional parameter uncertainties also varied along basin attributes (Figures 7, 8, and 10; Table 6). The clearest tercile examples were associated with snow-related uncertainty. In Figure 8, \(\mathrm{CFMAX}\) standard deviation decreased with snow fraction, with \(\rho=-0.92\) and a high-minus-low median difference of -0.04. \(\mathrm{TT}\) standard deviation also decreased with snow fraction, with \(\rho=-0.91\) and a high-minus-low difference of -0.09. Figure 10 shows similar declining snow-fraction gradients for snow-related uncertainty fields, including \(\mathrm{CFMAX}\), \(\mathrm{TT}\), and \(\mathrm{CWH}\). The uncertainty gradients in Figure 10 also included associations with aridity, slope, soil conductivity, vegetation, and soil-depth attributes.

The broader uncertainty matrix in Figure 7 shows that uncertainty–attribute relationships were structured but uneven in diagnostic clarity. The formal uncertainty-classification diagnostics identified \(\mathrm{CFMAX}\) and \(\mathrm{TT}\) as the less-confounded uncertainty cases among the selected gradients: \(\mathrm{CFMAX}\) standard deviation–snow fraction and \(\mathrm{TT}\) standard deviation–snow fraction had low mean–standard deviation coupling and low boundary sensitivity (Figure S15; Table S7). Other selected uncertainty gradients, including \(\mathrm{CWH}\) standard deviation–snow fraction, \(\mathrm{PERC}\) standard deviation–aridity, \(\mathrm{UZL}\) standard deviation–soil conductivity, and \(\mathrm{UH}_b\) standard deviation–snow fraction, were flagged as mean-coupled and boundary-sensitive. These rows were therefore retained as structured diagnostic patterns rather than interpreted as less-confounded parameter-scale uncertainty signals.

Representative basin cases illustrate how the large-sample gradients appear at the basin scale (Figure 11). The selected snow, arid, humid, steep, soil/storage, and routing-sensitive basins occupied distinct positions along the full-basin distributions of snow fraction, aridity, slope, soil conductivity, PET, and basin area (Figure 11b). Their parameter deviations from the all-basin median showed contrasts across snow, production-storage, recession, and routing parameter groups (Figure 11c). The within-parameter percentile profiles in Figure 11d further show that the representative basins differed across the full 14-parameter vector, rather than only in the parameter used to define each case. These basin-level profiles provide examples of how the matrix and gradient summaries appear in individual basins.

# 4 Discussion

## 4.1 Relationship reliability as an evaluation criterion

The results indicate that learned conceptual parameters require evaluation criteria beyond streamflow performance. All three formulations produced adequate simulations, and \(\delta_{\mathrm{dist}}\) did not clearly dominate NSE or KGE. If the study were framed only as a model-performance comparison, the evidence would be limited because the predictive distributions were broadly similar. Relationship reliability changes the target of comparison: it asks whether learned basin controls on conceptual parameters are reproducible enough to support interpretation within the adopted hydrologic model structure.

This criterion is particularly relevant for differentiable conceptual models. Neural regionalization can fit streamflow while using basin attributes in ways that are difficult to diagnose from predictive metrics alone. A stable parameter–attribute relationship is not automatically a direct physical relationship, but an unstable relationship provides a weak basis for hydrologic interpretation. In this study, the distributional formulation produced lower relationship variability across seeds and losses and a more compact relationship matrix. This supports its use for detailed gradient analysis, but it should be understood as a relationship-level result rather than a general statement of predictive superiority.

The deterministic and dropout baselines remain important in this framing. Both recovered several shared controls, and deterministic learning was competitive in prediction. The comparison therefore does not imply that the baselines are uninterpretable. Instead, it shows that formulation choice affects how reproducibly the learned relationship structure is recovered. Relationships that were reproduced across formulations, retained under seed and loss perturbations, and expressed as environmental gradients carry a different evidence status from relationships that appeared mainly in one formulation.

The additional parameter-value versus relationship-stability diagnostic reinforces this separation. Several parameters showed stable or low-variability values but less stable attribute–parameter relationships, often together with boundary saturation. Deterministic \(\mathrm{CFR}\), \(\mathrm{CWH}\), and \(\mathrm{UH}_a\), and dropout \(\mathrm{CFR}\), \(\mathrm{CWH}\), and \(\mathrm{K}_2\), are examples of this mismatch. These cases indicate that low parameter-value variability can arise from bounded transformations or restricted parameter ranges rather than from reproducible environmental controls. Thus, relationship reliability is not redundant with raw parameter stability.

The framework is intentionally pragmatic. It does not require accepting learned HBV parameters as direct physical measurements, and it does not require physical validation data for every parameter. Instead, it evaluates whether the relationships used for interpretation persist across a set of reasonable training and formulation perturbations. This evidence-chain view also reduces over-reliance on single visual summaries. Spatial maps, correlation matrices, and gradient plots can each appear organized, but their interpretive value depends on whether the same relationships are stable across seeds, less sensitive to losses, not dominated by boundaries, and not confined to a single learning formulation.

## 4.2 Hydrologic interpretation under attribute collinearity

The shared dominant-control core provides the main hydrologic information in the learned parameter fields. The most interpretable relationships linked \(\mathrm{BETA}\) with basin slope, \(\mathrm{FC}\) with potential evapotranspiration, \(\mathrm{PERC}\) with aridity, and \(\mathrm{UZL}\) with soil conductivity. These relationships were shared across formulations, had relatively low seed sensitivity in the distributional formulation, and appeared as environmental gradients in the distributional parameter means. They can therefore be discussed as reproducible behavioral gradients within the differentiable HBV framework.

The hydrologic interpretation remains behavioral rather than causal. The negative \(\mathrm{BETA}\)–slope relationship indicates that the learned runoff-generation nonlinearity varied along a terrain gradient. The positive \(\mathrm{FC}\)–PET relationship indicates that the learned storage-capacity parameter increased along an evaporative-demand gradient. The negative \(\mathrm{PERC}\)–aridity relationship indicates that percolation-related behavior differed across wet–dry conditions. The positive \(\mathrm{UZL}\)–soil conductivity relationship links upper-zone threshold behavior with soil hydraulic descriptors. These patterns are consistent with the process roles of the HBV parameters, but they should not be read as direct measurements of isolated physical properties.

The attribute-collinearity analysis clarifies this interpretation. Several key attributes were correlated across the 531 basins (Figure S13; Table S5). The largest correlations included high-precipitation frequency with low-precipitation frequency (\(\rho=0.939\)), aridity with mean precipitation (\(\rho=-0.867\)), and mean slope with mean elevation (\(\rho=0.807\)). Additional correlations included slope with soil depth (\(\rho=-0.798\)), elevation with snow fraction (\(\rho=0.687\)), aridity with PET (\(\rho=0.600\)), and slope with precipitation seasonality (\(\rho=-0.602\)). These correlations mean that dominant attributes should be interpreted as representatives of broader hydroclimatic or physiographic gradients, not as isolated causal controls.

Residual sensitivity checks provided a more specific diagnostic for the four main shared relationships (Table S6). After rank-residualizing against selected correlated attributes, the relationship directions were retained for all four pairs. The \(\mathrm{BETA}\)–slope relationship changed from \(\rho=-0.523\) to a partial residual \(\rho=-0.335\) after controlling for elevation and snow fraction. The \(\mathrm{FC}\)–PET relationship changed from \(\rho=0.487\) to \(\rho=0.281\) after controlling for aridity and mean precipitation. The \(\mathrm{PERC}\)–aridity relationship changed from \(\rho=-0.481\) to \(\rho=-0.221\) after controlling for PET and mean precipitation. The \(\mathrm{UZL}\)–soil conductivity relationship changed from \(\rho=0.534\) to \(\rho=0.460\) after controlling for soil depth, slope, and forest fraction. These checks do not isolate causal effects, but they indicate that the main relationship directions were not fully explained by the selected correlated descriptors.

The \(\mathrm{PERC}\)–aridity relationship showed the largest weakening after residualization and should therefore be interpreted more cautiously than the \(\mathrm{UZL}\)–soil conductivity relationship. This difference is consistent with the collinearity structure, because aridity was strongly related to mean precipitation and moderately related to PET, low-precipitation frequency, forest fraction, and other descriptors. In contrast, soil conductivity had fewer high-correlation counterparts in the selected attribute set. The evidence hierarchy therefore supports a graded interpretation: some relationships retain both direction and moderate magnitude after residualization, whereas others retain direction but are more clearly embedded in correlated environmental gradients.

Snow-related gradients also illustrate this distinction. \(\mathrm{CWH}\) and \(\mathrm{CFR}\) had large negative relationships with snow fraction in the distributional formulation, but their dominant controls were partially shared rather than fully shared across formulations. Snow fraction itself was correlated with elevation, and snow-related parameter behavior can also interact with temperature thresholds, precipitation timing, and terrain. These gradients are therefore useful as distributional diagnostic patterns of learned snow-process behavior, while their cross-formulation status remains more cautious than that of the four main shared controls.

Routing and recession parameters require similar caution. Several of these parameters showed spatial organization and attribute dependence, but their dominant controls were less consistent than the main production and storage relationships. Routing and recession parameters can compensate for timing errors, storage partitioning, snowmelt representation, or forcing uncertainty in a conceptual model. The shared relationship involving \(\mathrm{UH}_a\) is therefore better described as a reproducible routing-related diagnostic than as a direct geomorphic parameter. The less consistent behavior of \(\mathrm{UH}_b\) supports treating some routing controls as formulation-sensitive.

## 4.3 Hydroclimatic-group sensitivity and regional interpretation

The hydroclimatic-group analyses provide an additional check on whether the main relationships were dominated by one subset of basins. In the leave-one-group-out analysis, none of the main distributional relationships changed by \(|\Delta \rho| \ge 0.15\) when any single hydroclimatic group was removed (Figure S14a; Table S4). The leave-one-group-out correlations were close to the full-sample values for \(\mathrm{BETA}\)–slope, \(\mathrm{FC}\)–PET, \(\mathrm{PERC}\)–aridity, \(\mathrm{UZL}\)–soil conductivity, \(\mathrm{CWH}\)–snow fraction, and \(\mathrm{CFR}\)–snow fraction. This result indicates that the full-sample relationships were not controlled by one hydroclimatic stratum alone.

The within-group correlations showed larger heterogeneity (Figure S14b; Table S4). Some relationships weakened within individual groups, and a small number changed sign. For example, the \(\mathrm{FC}\)–PET relationship was positive in several groups but became negative in the snow arid steep group, and the \(\mathrm{PERC}\)–aridity relationship changed sign in the low-snow arid steep group. The \(\mathrm{UZL}\)–soil conductivity relationship remained positive in the groups shown but varied in magnitude, and the \(\mathrm{CFR}\)–snow fraction relationship retained a negative sign while showing different group-specific strengths.

These within-group differences are consistent with the narrower environmental range of each hydroclimatic stratum. A relationship that is visible across the full continental sample may be weaker within a restricted hydroclimatic group if the relevant environmental gradient is compressed. Group-specific compensation can also occur within the HBV structure. The group-wise diagnostics therefore support a moderate interpretation: the full-sample gradients are not driven by one group alone, but the same relationship magnitude should not be assumed within every hydroclimatic setting.

These results are not an ungauged-basin transfer validation. The models were not retrained under leave-region-out designs, and the group-wise analyses do not test predictive transfer to unseen regions. They instead provide a sensitivity check on the relationship summaries. For regional interpretation, this distinction matters. The learned relationships can be discussed as broad behavioral gradients across the 531 CAMELS-US basins, while more specific claims about ungauged transfer would require a separate experimental design.

## 4.4 Diagnostic role of distributional parameter uncertainty

The distributional formulation provides parameter-scale spread in addition to parameter means. In the Results, several parameter standard deviations varied along basin attributes. The selected examples with the lowest diagnostic concern were snow-related uncertainty gradients, especially the decreases in \(\mathrm{CFMAX}\) and \(\mathrm{TT}\) standard deviations with snow fraction. These gradients were visible in the tercile summaries and detailed gradient panels and were classified as less confounded in the formal uncertainty diagnostics.

The uncertainty classification shows that only a subset of uncertainty gradients should be interpreted with relatively low diagnostic concern. \(\mathrm{CFMAX}\) standard deviation–snow fraction had \(\rho=-0.9185\), low mean–standard deviation coupling (\(\rho=0.042\)), low boundary-distance coupling (\(\rho=0.039\)), and a near-boundary share of 0.147. \(\mathrm{TT}\) standard deviation–snow fraction had \(\rho=-0.9105\), mean–standard deviation coupling of \(-0.181\), boundary-distance coupling of \(-0.176\), and a near-boundary share of 0.009. In Figure S15, these diagnostics are shown using absolute correlation magnitudes, \(|\rho(\mathrm{mean}, \mathrm{std})|\) and \(|\rho(\mathrm{boundary\ distance}, \mathrm{std})|\), so that both positive and negative coupling indicate stronger diagnostic concern. Under this classification, \(\mathrm{CFMAX}\) and \(\mathrm{TT}\) fall in the less-confounded region.

Other uncertainty gradients were more affected by parameter means or boundary proximity. \(\mathrm{CWH}_{std}\)–snow fraction had a large correlation with snow fraction, but \(\mathrm{CWH}\) also had high mean–standard deviation coupling and high boundary-distance coupling, together with a near-boundary share above 0.5. \(\mathrm{PERC}_{std}\)–aridity, \(\mathrm{UZL}_{std}\)–soil conductivity, and \(\mathrm{UH}_{b,std}\)–snow fraction were also classified as mean-coupled and boundary-sensitive. These relationships may still contain structured information about the learned distributional mapping, but they cannot be interpreted in the same way as the less-confounded \(\mathrm{CFMAX}\) and \(\mathrm{TT}\) uncertainty gradients.

This distinction is important because a predicted distributional standard deviation is not a calibrated Bayesian posterior. It depends on the neural parameterization, the bounded transformation, the training objective, the HBV model structure, and the information available in static attributes and streamflow. The diagnostic flags therefore constrain how uncertainty fields should be used. Relationships with low coupling and low boundary sensitivity can be described as less-confounded parameter-scale spread patterns, while relationships in the coupled or boundary-sensitive regions should remain diagnostic rather than being treated as evidence of physical parameter identifiability.

For regional hydrologic modeling, this diagnostic layer is still useful. A point-estimate regionalization method produces one parameter vector per basin, whereas the distributional formulation also provides a parameter-spread field that can be screened against basin attributes, parameter means, and boundary distances. This additional information can help identify where the learned mapping is more variable and which uncertainty gradients require caution. Its value is therefore conditional: it is most informative when interpreted together with relationship stability, parameter means, and boundary diagnostics.

## 4.5 Implications, limitations, and future applications

The proposed evidence hierarchy is relevant to regionalized hydrologic modeling because regionalization assumes that basin attributes contain transferable information about model parameters. Differentiable parameter learning makes this assumption explicit by training an attribute-to-parameter mapping. The present results suggest that this mapping should be evaluated not only by streamflow prediction, but also by the reproducibility of its learned parameter–attribute relationships.

For ungauged-basin reasoning, the implication is methodological rather than operational. This study did not perform a separate ungauged-basin transfer experiment, and the results should not be interpreted as evidence that the same relationships will necessarily hold under spatial extrapolation. However, relationship reliability can serve as a screening step before such transfer analyses. Relationships that remain stable across seeds and losses, appear across formulations, and are visible as environmental gradients provide a clearer basis for regional interpretation than relationships that are sensitive to training perturbations.

The representative basin cases provide examples of how large-sample gradients appear in individual basins. The selected snow, arid, humid, steep, soil/storage, and routing-sensitive basins occupied different positions along snow fraction, aridity, slope, soil conductivity, PET, and area gradients. Their parameter profiles differed across snow, production-storage, recession, and routing parameter groups. These cases help connect the matrix and gradient summaries to basin-scale examples, but they are illustrative rather than independent validation evidence.

Several limitations constrain the interpretation. First, the analysis uses one HBV-type conceptual model. HBV parameters are behavioral summaries of model processes and are affected by the chosen model structure, parameter bounds, and routing representation. Other conceptual models or different parameter ranges could produce different learned relationships. Second, the analysis uses CAMELS-US basins and the available CAMELS static attributes. The inferred gradients may depend on this basin sample, the forcing data, the attribute set, and the training and testing periods.

Third, the perturbation design covers random seeds and three loss functions, but not all sources of uncertainty. It does not include forcing uncertainty, observation uncertainty, alternative model structures, alternative neural-network architectures, or different calibration periods. The deduplication and sensitivity checks support the reported relationship-level comparisons within the current experiment, but they do not make the learned parameter values physical observations. Fourth, the analysis is cross-basin and associative. Spearman correlations, tercile contrasts, residualized correlations, and gradient panels quantify empirical relationships between basin descriptors and learned parameters. They do not isolate causal effects of individual attributes.

Future work should test whether the relationship-reliability framework generalizes to other model structures, other large-sample data sets, and explicit ungauged-basin splits. Additional experiments could compare distributional parameter spread with independent process observations, posterior inference, or ensemble-based uncertainty estimates. Such tests would help clarify when learned parameter uncertainty reflects information limitations, structural compensation, or boundary effects. Within the present study, the uncertainty field is best viewed as structured diagnostic information produced by the distributional parameter-learning model.

# 5 Conclusions

This study evaluated deterministic, Monte Carlo dropout, and distributional formulations for learning basin-specific static HBV parameters from CAMELS-US basin attributes in a differentiable hydrologic modeling framework. Predictive metrics were used to establish simulation adequacy, rather than to select an interpretive formulation. All three formulations produced adequate simulations, and the distributional formulation did not clearly outperform the deterministic or dropout baselines in median NSE or KGE.

The main contribution is a relationship-reliability framework for learned conceptual hydrologic parameters. The framework evaluates whether basin attribute–parameter relationships are stable across random seeds, loss functions, and plausible learning formulations before they are used for hydrologic interpretation. Under this framework, \(\delta_{\mathrm{dist}}\) provided the most reproducible relationship structure among the three tested formulations, with lower cross-seed and cross-loss variability in Spearman correlations, higher dominant-control reproducibility, and higher matrix-level compactness. This result indicates a structure-level reliability advantage, not a claim that every parameter value or every individual relationship is more stable.

The hydrologic interpretation is selective. Seven of the fourteen HBV parameters formed a shared dominant-control core across formulations. The clearest examples linked \(\mathrm{BETA}\) with basin slope, \(\mathrm{FC}\) with potential evapotranspiration, \(\mathrm{PERC}\) with aridity, and \(\mathrm{UZL}\) with soil conductivity. These relationships combined cross-formulation agreement, seed and loss stability, and visible environmental gradients. Snow-related parameters also showed high-magnitude distributional gradients with snow fraction, but those gradients were partially shared and more formulation-dependent, so they are treated as behavioral diagnostic evidence rather than general parameter rules.

Distributional parameter uncertainty added a useful but limited diagnostic layer. Some uncertainty gradients, especially for snow-process parameters, were less affected by the mean–standard deviation coupling and boundary-distance diagnostics used here. Other high-magnitude uncertainty gradients were flagged as mean-coupled or boundary-sensitive. The uncertainty field should therefore be interpreted as structured information about the learned mapping, not as calibrated posterior uncertainty or direct evidence of physical parameter identifiability.

The practical conclusion is that learned conceptual parameters should pass multiple evidence filters before supporting hydrologic inference. Adequate streamflow simulation is necessary but insufficient. Key relationships should also be stable across seeds, reasonably insensitive to objective-function choice, consistent with alternative formulations when possible, visible in basin gradients, and screened for parameter-bound or uncertainty-coupling artifacts. Relationships that satisfy these checks can support cautious regional interpretation; relationships that fail them can remain useful diagnostics but should not be presented as reproducible basin-control relationships.

# Table

---

## Table 1

**Table 1. Predictive performance of the three parameter-learning formulations across 531 CAMELS-US basins.**

| Summary level     | Metric                       | \(\delta_{\mathrm{base}}\) | \(\delta_{\mathrm{mcd}}\) | \(\delta_{\mathrm{dist}}\) |
| ----------------- | ---------------------------- | ------------------------:| -----------------------:| ------------------------:|
| All complete runs | Median NSE                   | 0.611                    | 0.624                   | 0.611                    |
| All complete runs | Median KGE                   | 0.626                    | 0.639                   | 0.625                    |
| Hybrid loss       | Median NSE                   | 0.632                    | 0.636                   | 0.632                    |
| Hybrid loss       | Median KGE                   | 0.675                    | 0.679                   | 0.675                    |
| All complete runs | Fraction NSE (>0.5)          | 0.74                     | 0.74                    | 0.73                     |
| All complete runs | Fraction KGE (>0.5)          | 0.83                     | 0.81                    | 0.83                     |
| All complete runs | Median absolute bias         | 0.736                    | 0.790                   | 0.732                    |
| All complete runs | Median absolute percent bias | 46.397                   | 47.384                  | 46.497                   |

**Table note:** NSE, KGE, absolute bias, and absolute percent bias were computed basin by basin over the test period after warm-up removal and then summarized across complete runs. The hybrid-loss rows correspond to the reference loss setting used in Figure 1.

---

## Table 2 placeholder

**Table 2. Relationship reliability diagnostics across formulations.**

| Metric                                                | \(\delta_{\mathrm{base}}\) | \(\delta_{\mathrm{mcd}}\) | \(\delta_{\mathrm{dist}}\) |
| ----------------------------------------------------- | ------------------------:| -----------------------:| ------------------------:|
| Median cross-seed SD of Spearman (\rho)               | 0.0461                   | 0.0347                  | 0.0241                   |
| Median cross-seed range of Spearman (\rho)            | 0.1122                   | 0.0891                  | 0.0630                   |
| Mean cross-seed SD of Spearman (\rho)                 | 0.0663                   | 0.0546                  | 0.0308                   |
| Median dominant-attribute consistency                 | 0.800                    | 0.800                   | 1.000                    |
| Median top-5 overlap                                  | 0.629                    | 0.679                   | 0.700                    |
| Mean cross-loss SD of Spearman (\rho)                 | 0.142                    | 0.168                   | 0.129                    |
| Median hydroclimatic-group seed SD of Spearman (\rho) | 0.0586                   | 0.0541                  | 0.0491                   |

**Table note:** Candidate relationships were selected using mean absolute Spearman correlation across all formulations, losses, and seeds. Cross-seed metrics evaluate relationship reproducibility under different random initializations. Cross-loss metrics evaluate sensitivity to hydrologic objective functions. Top-5 overlap is computed as the Jaccard overlap among the five strongest attribute–parameter relationships for each parameter.

---

## Table 3

**Table 3. Dominant basin controls and cross-formulation consistency for the 14 learned HBV parameters.**

| Parameter        | Dominant control in \(\delta_{\mathrm{base}}\) | Dominant control in \(\delta_{\mathrm{mcd}}\) | Dominant control in \(\delta_{\mathrm{dist}}\) | Direction in \(\delta_{\mathrm{dist}}\) | \(\rho_{\mathrm{dist}}\) | Consistency class                   |
| ---------------- | -------------------------------------------- | ------------------------------------------- | -------------------------------------------- | ------------------------------------- | ----------------------:| ----------------------------------- |
| \(\mathrm{BETA}\)  | slope_mean                                   | slope_mean                                  | slope_mean                                   | −                                     | -0.583                 | Shared                              |
| \(\mathrm{FC}\)    | pet_mean                                     | pet_mean                                    | pet_mean                                     | +                                     | 0.509                  | Shared                              |
| \(\mathrm{LP}\)    | high_prec_dur                                | lai_diff                                    | p_seasonality                                | −                                     | -0.187                 | Model-sensitive                     |
| \(\mathrm{PERC}\)  | aridity                                      | aridity                                     | aridity                                      | −                                     | -0.594                 | Shared                              |
| \(\mathrm{UZL}\)   | soil_conductivity                            | soil_conductivity                           | soil_conductivity                            | +                                     | 0.570                  | Shared                              |
| \(\mathrm{K}_0\)   | frac_snow                                    | soil_conductivity                           | frac_snow                                    | −                                     | -0.424                 | Partially shared                    |
| \(\mathrm{K}_1\)   | high_prec_dur                                | gvf_diff                                    | high_prec_dur                                | −                                     | -0.454                 | Partially shared                    |
| \(\mathrm{K}_2\)   | high_prec_dur                                | high_prec_dur                               | high_prec_dur                                | −                                     | -0.336                 | Shared                              |
| \(\mathrm{TT}\)    | elev_mean                                    | elev_mean                                   | elev_mean                                    | +                                     | 0.265                  | Shared                              |
| \(\mathrm{CFMAX}\) | clay_frac                                    | frac_snow                                   | clay_frac                                    | −                                     | -0.426                 | Partially shared                    |
| \(\mathrm{CFR}\)   | slope_mean                                   | frac_snow                                   | frac_snow                                    | −                                     | -0.625                 | Partially shared                    |
| \(\mathrm{CWH}\)   | slope_mean                                   | frac_snow                                   | frac_snow                                    | −                                     | -0.904                 | Partially shared                    |
| \(\mathrm{UH}_a\)  | slope_mean                                   | slope_mean                                  | slope_mean                                   | −                                     | -0.426                 | Shared                              |
| \(\mathrm{UH}_b\)  | aridity                                      | frac_snow                                   | aridity                                      | −                                     | -0.367                 | Partially shared; sign-inconsistent |

**Table note:** Dominant controls are modal dominant attributes across losses and seeds within each formulation. The distributional direction and \(\rho_{\mathrm{dist}}\) are the Spearman correlation for the listed \(\delta_{\mathrm{dist}}\) control in the distributional mean relationship matrix. “Shared” indicates consistent dominant control and direction across all three formulations; “partially shared” indicates agreement in two formulations or related but not identical dominant controls; “model-sensitive” indicates inconsistent dominant controls across formulations.

---

## Table 4

**Table 4. Matrix-level consistency of attribute–parameter relationship structures across formulations.**

| Metric                                             | \(\delta_{\mathrm{base}}\) | \(\delta_{\mathrm{mcd}}\) | \(\delta_{\mathrm{dist}}\) | Interpretation                                                                      |
| -------------------------------------------------- | ------------------------:| -----------------------:| ------------------------:| ----------------------------------------------------------------------------------- |
| Within-formulation compactness                     | 0.702                    | 0.680                   | 0.809                    | Larger values indicate more internally consistent relationship matrices             |
| Mean within-formulation Frobenius distance         | 3.453                    | 3.668                   | 2.624                    | Smaller values indicate more similar relationship matrices                          |
| Mean within-formulation cosine similarity          | 0.701                    | 0.679                   | 0.808                    | Larger values indicate closer matrix orientation                                    |
| Mean intermodel similarity with other formulations | 0.682                    | 0.677                   | 0.679                    | Used to assess whether a formulation departs from the shared relationship structure |
| Number of shared dominant controls                 | 7                        | 7                       | 7                        | Summary of parameter-level agreement                                                |
| Number of partially shared controls                | 6                        | 6                       | 6                        | Summary of formulation-sensitive agreement                                          |
| Number of model-sensitive controls                 | 1                        | 1                       | 1                        | Summary of unstable dominant controls                                               |

**Table note:** Compactness is computed as the mean within-formulation pairwise matrix correlation across runs. Frobenius distance is computed between full attribute-by-parameter Spearman correlation matrices. Lower Frobenius distance and higher compactness indicate a more internally consistent relationship structure. Intermodel similarity is used to evaluate whether the distributional formulation remains within the shared basin-control structure rather than forming a separate relationship regime.

## Table 5. Selected robust distributional mean gradients

High-minus-low values are tercile median differences on the normalized parameter scale.

| Parameter | Attribute         | Spearman rho | High-minus-low tercile median difference | Evidence role                                      |
| --------- | ----------------- | ------------:| ----------------------------------------:| -------------------------------------------------- |
| \(\mathrm{BETA}\) | Mean slope        | -0.583       | -0.403                                   | Shared dominant-control evidence                   |
| \(\mathrm{FC}\)   | Mean PET          | 0.509        | 0.198                                    | Shared dominant-control evidence                   |
| \(\mathrm{PERC}\) | Aridity           | -0.594       | -0.229                                   | Shared dominant-control evidence                   |
| \(\mathrm{UZL}\)  | Soil conductivity | 0.570        | 0.260                                    | Shared dominant-control evidence                   |
| \(\mathrm{CWH}\)  | Snow fraction     | -0.904       | -0.343                                   | Strong but partially shared snow-gradient evidence |
| \(\mathrm{CFR}\)  | Snow fraction     | -0.625       | -0.076                                   | Strong but partially shared snow-gradient evidence |

## Table 6. Selected uncertainty gradients and diagnostic flags

Uncertainty gradients use normalized distributional parameter standard deviation. High-minus-low values are included where available from the key-gradient panels; otherwise entries refer to matrix or detailed-gradient evidence.

| Parameter uncertainty | Attribute         | Spearman rho | High-minus-low tercile median difference | Diagnostic flag        |
| --------------------- | ----------------- | ------------:| ----------------------------------------:| ---------------------- |
| \(\mathrm{CFMAX}\) | Snow fraction     | -0.919       | -0.040                                   | Less confounded        |
| \(\mathrm{TT}\)    | Snow fraction     | -0.910       | -0.089                                   | Less confounded        |
| \(\mathrm{CWH}\)   | Snow fraction     | -0.922       | Detailed gradient in Figure 10           | Interpret with caution |
| \(\mathrm{PERC}\)  | Aridity           | -0.584       | Detailed gradient in Figure 10           | Interpret with caution |
| \(\mathrm{UZL}\)   | Soil conductivity | 0.578        | Detailed gradient in Figure 10           | Interpret with caution |
| \(\mathrm{UH}_b\)  | Snow fraction     | 0.599        | Matrix diagnostic in Figure 7            | Interpret with caution |

# Figure Captions

**Figure 1. Predictive performance of the three parameter-learning formulations.** Panels (a) and (b) compare basin-level NSE and KGE for \(\delta_{\mathrm{base}}\), \(\delta_{\mathrm{mcd}}\), and \(\delta_{\mathrm{dist}}\) under the reference hybrid loss, with medians printed inside each boxplot. Panels (c) and (d) show empirical CDFs and the fractions of basins exceeding NSE or KGE of 0.5. The three formulations have similar performance distributions, so the metrics are used as adequacy checks for the 531-basin analysis rather than as the basis for selecting an interpretive formulation.

**Figure 2. Cross-seed parameter stability, boundary saturation, and stochastic interval behavior.** Panels (a1)-(a2) summarize normalized seed variability by parameter and formulation, including pooled summaries before and after excluding boundary-sensitive parameters. Panels (b1)-(b2) show boundary saturation and distance-to-boundary diagnostics, indicating where low apparent variability can partly reflect bounded transformations. Panels (c1)-(c2) compare q05-q95 interval widths for the stochastic formulations and representative parameter-sample distributions. Raw parameter stability is therefore interpreted together with boundary behavior and interval width.

**Figure 3. Cross-seed stability of basin attribute-parameter relationships.** Candidate high-magnitude relationships are selected by mean absolute Spearman correlation across all formulations, losses, and seeds, avoiding selection based on \(\delta_{\mathrm{dist}}\) alone. Panels summarize pooled seed SD, parameter-level seed variability, sign consistency, dominant-control consistency, top-5 overlap, and four representative robust pairs. Lower seed SD and narrower seed ranges indicate more reproducible relationship recovery, but do not by themselves prove physical parameter truth.

**Figure 4. Basin-group relationship stability.** Panel (a) maps the 531 CAMELS-US basins in seven hydroclimatic groups used for robustness checks. Panels (b)-(d) compare group-wise seed variability in high-correlation relationships, top-5 relationship overlap, and normalized within-group parameter spread. The figure evaluates whether relationship reliability persists across major basin environments; the groups are robustness strata rather than independent validation experiments.

**Figure 5. Spatial organization of distributional HBV parameter means.** Maps show seed-averaged \(\delta_{\mathrm{dist}}\) means for 14 HBV and routing parameters across the 531 CAMELS-US basins under the reference hybrid loss. Each panel uses the physical search range of its own parameter, so color scales should be read within panels rather than compared directly across parameters. The maps provide a descriptive spatial check on the learned field and motivate the subsequent attribute-gradient analyses; they are not a standalone spatial significance test.

**Figure 6. Distributional parameter-mean relationships with basin attributes.** Panel (a) is a 35-attribute by 14-parameter circle heatmap of Spearman correlations between \(\delta_{\mathrm{dist}}\) parameter means and basin attributes. Color gives signed correlation and circle size gives absolute magnitude; black dots mark strong relationships, stars mark stable strong relationships, and boxes mark the dominant attribute for each parameter. Panels (b) and (c) summarize top-ranked relationship strength by parameter and by attribute group, linking the matrix view to hydrologic process categories.

**Figure 7. Distributional parameter-uncertainty relationships and diagnostics.** Panel (a) shows Spearman correlations between normalized \(\delta_{\mathrm{dist}}\) parameter standard deviations and basin attributes, using the same color and size encoding as the mean-relationship matrix. Panel (b) summarizes parameter-level uncertainty-structure strength, and panel (c) compares mean-standard deviation coupling with boundary sensitivity. Grey caution markers identify strong uncertainty relationships affected by these diagnostics, so the figure treats uncertainty gradients as structured diagnostic signals rather than calibrated posterior uncertainty.

**Figure 8. Key environmental gradients in distributional parameter means and uncertainties.** Panels (a)-(f) show selected mean gradients for snow fraction, aridity, slope, soil conductivity, and potential evapotranspiration; panels (g)-(l) show selected uncertainty gradients. Basins are grouped into low, middle, and high attribute terciles, with jittered points and boxplots showing within-group distributions. Annotations report Spearman rho, FDR-adjusted q values, and high-minus-low median differences. These tercile contrasts summarize basin-scale gradients and are not causal tests.

**Figure 9. Detailed distributional parameter-mean gradients.** Twenty selected attribute-parameter mean relationships are organized by process group: snow and seasonality, aridity and evapotranspiration, terrain and topography, soil and storage, and routing or extremes. Grey points are individual basins, colored lines show quantile-binned medians, and shaded bands show interquartile ranges. Panel annotations give mean Spearman rho and seed SD for each pair, connecting gradient shape with relationship stability.

**Figure 10. Detailed distributional parameter-uncertainty gradients.** The layout mirrors Figure 9 but uses \(\delta_{\mathrm{dist}}\) parameter standard deviation, averaged over seeds and losses, as the response variable. Panels show selected uncertainty-attribute relationships across the same process groups using basin points, quantile-binned medians, and interquartile ranges. These gradients show where learned parameter spread is environmentally structured, but interpretation depends on the coupling and boundary diagnostics summarized in Figure 7.

**Figure 11. Representative basin cases and parameter regimes.** Panel (a) maps the selected snow, arid, humid, steep, soil/storage, and routing-sensitive case basins. Panel (b) reports attribute percentile ranks for the same cases. Panel (c) compares parameter-mean deviations from the all-basin median, with circle size indicating normalized parameter uncertainty. Panel (d) shows selected case-specific parameter profiles and their NSE and KGE values. The cases illustrate how large-sample gradients appear in individual basins and are not used as independent inferential evidence.

# Appendix

## Appendix A. Training inventory, data readiness, and sensitivity checks

The final run inventory contained 45 complete runs: three model formulations, three loss functions, and five random seeds. The formulations were deterministic \(\delta_{\mathrm{base}}\), Monte Carlo dropout \(\delta_{\mathrm{mcd}}\), and distributional \(\delta_{\mathrm{dist}}\). The loss functions were `HybridNseBatchLoss`, `LogNseBatchLoss`, and `NseBatchLoss`. Seeds were 111, 222, 333, 444, and 555. Each complete parameter run contained 531 basins and 14 parameters after run-level processing.

Raw parameter source files contained duplicate logical rows for some model/loss/seed/basin/parameter keys. The pipeline diagnosed duplicate keys and collapsed them by averaging within each logical key. Duplicate-key diagnostics identified 200,718 duplicate logical parameter keys, including 162,691 keys with mean or standard-deviation conflicts. This issue is handled before the main relationship analyses.

Deduplication sensitivity checks show that the main relationship-reliability ordering is stable. For \(\delta_{\mathrm{dist}}\), mean cross-seed SD of Spearman rho changed from 0.03120 in the raw-as-is table to 0.03131 after deduplication, and mean cross-loss SD changed from 0.13075 to 0.13132. Core pairwise relationships also changed little after deduplication: distributional \(\mathrm{BETA}\)-slope_mean changed by -0.00048 in rho, \(\mathrm{FC}\)-pet_mean by 0.00030, \(\mathrm{PERC}\)-aridity by -0.00041, \(\mathrm{UZL}\)-soil_conductivity by 0.00131, \(\mathrm{CWH}\)-frac_snow by -0.00191, and \(\mathrm{CFR}\)-frac_snow by -0.00826. These checks support the use of deduplicated tables for relationship-level inference.

Attribute collinearity was summarized for the main environmental gradients. Spearman correlations among selected attributes included slope-elevation = 0.807, snow fraction-elevation = 0.687, snow fraction-slope = 0.568, aridity-PET = 0.600, and slope-precipitation seasonality = -0.602. These correlations support the interpretation of dominant attributes as stable empirical gradients rather than isolated causal controls.

## Appendix B. Parameter bounds and normalized quantities

## Table S1. HBV and routing parameter definitions and bounds.

| Parameter | Process group | Hydrologic role | Unit | Lower bound | Upper bound | Normalized quantity | Interpretation note |
|---|---|---|---|---:|---:|---|---|
| \(\mathrm{BETA}\) | Soil/runoff generation | Controls nonlinearity of runoff generation from soil moisture storage | - | 1.0 | 6.0 | \((\theta-L)/(U-L)\) | Behavioral runoff-generation parameter |
| \(\mathrm{FC}\) | Soil moisture | Maximum soil water storage capacity | mm | 50 | 1000 | \((\theta-L)/(U-L)\) | Behavioral storage-capacity parameter |
| \(\mathrm{LP}\) | Evapotranspiration | Soil-moisture threshold controlling potential evapotranspiration limitation | - | 0.2 | 1.0 | \((\theta-L)/(U-L)\) | Behavioral evapotranspiration-limitation parameter |
| \(\mathrm{PERC}\) | Percolation | Maximum percolation from upper to lower groundwater storage | mm d\(^{-1}\) | 0 | 10 | \((\theta-L)/(U-L)\) | Behavioral percolation parameter |
| \(\mathrm{UZL}\) | Upper groundwater | Upper-zone threshold for fast runoff response | mm | 0 | 100 | \((\theta-L)/(U-L)\) | Behavioral upper-zone threshold parameter |
| \(\mathrm{K}_0\) | Recession | Fast runoff recession coefficient | d\(^{-1}\) | 0.05 | 0.9 | \((\theta-L)/(U-L)\) | Behavioral fast-recession parameter |
| \(\mathrm{K}_1\) | Recession | Intermediate runoff recession coefficient | d\(^{-1}\) | 0.01 | 0.5 | \((\theta-L)/(U-L)\) | Behavioral recession parameter |
| \(\mathrm{K}_2\) | Recession | Slow/baseflow recession coefficient | d\(^{-1}\) | 0.001 | 0.2 | \((\theta-L)/(U-L)\) | Behavioral slow-recession parameter |
| \(\mathrm{TT}\) | Snow | Temperature threshold separating rainfall and snowfall / melt conditions | \(^{\circ}\mathrm{C}\) | -2.5 | 2.5 | \((\theta-L)/(U-L)\) | Behavioral snow-threshold parameter |
| \(\mathrm{CFMAX}\) | Snow | Degree-day snowmelt factor | mm \(^{\circ}\mathrm{C}^{-1}\) d\(^{-1}\) | 0.5 | 10 | \((\theta-L)/(U-L)\) | Behavioral snowmelt parameter |
| \(\mathrm{CFR}\) | Snow | Refreezing coefficient | - | 0 | 0.1 | \((\theta-L)/(U-L)\) | Behavioral refreezing parameter |
| \(\mathrm{CWH}\) | Snow | Liquid water holding capacity of snowpack | - | 0 | 0.2 | \((\theta-L)/(U-L)\) | Behavioral snowpack-retention parameter |
| \(\mathrm{UH}_a\) | Routing | Shape parameter of gamma unit hydrograph | - | 0 | 2.9 | \((\theta-L)/(U-L)\) | Behavioral routing parameter |
| \(\mathrm{UH}_b\) | Routing | Scale parameter of gamma unit hydrograph | d | 0 | 6.5 | \((\theta-L)/(U-L)\) | Behavioral routing parameter |

**Table note.** Parameters are reported using the notation adopted in the manuscript. Bounds are used for bounded parameter transformation and normalization. All parameters are interpreted as behavioral parameters of the adopted differentiable HBV model, not as direct observations of physical properties.

Parameter means and standard deviations are normalized by these ranges when compared across parameters.

## Table S2. Model inputs, streamflow target, static basin attributes, and preprocessing.

| Data element | Variables or fields | Role in the analysis | Preprocessing and use |
|---|---|---|---|
| Dynamic meteorological forcing | Daily precipitation, mean air temperature, and potential evapotranspiration | Drives the differentiable HBV simulation for each basin and day | Extracted from CAMELS-US forcing records for the 531-basin study subset. The HBV simulation uses the daily forcing sequence with a 365-day warm-up period before metric calculation. |
| Streamflow target | Daily observed streamflow | Training target and basis for NSE, KGE, absolute bias, and absolute percent bias | Observed discharge is converted to basin-area-normalized depth units before training and evaluation. Metrics are computed on the test period after warm-up removal. |
| Static hydroclimatic attributes | p_mean, pet_mean, p_seasonality, frac_snow, aridity, high_prec_freq, high_prec_dur, low_prec_freq, low_prec_dur | Basin descriptors used by the attribute-to-parameter network and by relationship-reliability analyses | Static attributes are basin-level quantities. They enter the neural parameterization network only, not the HBV time-stepping equations, and are used for cross-basin attribute-parameter correlations. |
| Static topographic and basin-size attributes | elev_mean, slope_mean, area_gages2 | Basin descriptors used by the attribute-to-parameter network and by relationship-reliability analyses | Values are matched to the 531 retained basins and treated as time-invariant descriptors. |
| Static vegetation and land-cover attributes | frac_forest, lai_max, lai_diff, gvf_max, gvf_diff, dom_land_cover_frac, dom_land_cover, root_depth_50 | Basin descriptors used by the attribute-to-parameter network and by relationship-reliability analyses | Values are matched to the 531 retained basins and treated as time-invariant descriptors. |
| Static soil attributes | soil_depth_pelletier, soil_depth_statsgo, soil_porosity, soil_conductivity, max_water_content, sand_frac, silt_frac, clay_frac | Basin descriptors used by the attribute-to-parameter network and by relationship-reliability analyses | Values are matched to the 531 retained basins and treated as time-invariant descriptors. |
| Static geology attributes | geol_1st_class, glim_1st_class_frac, geol_2nd_class, glim_2nd_class_frac, carbonate_rocks_frac, geol_porosity, geol_permeability | Basin descriptors used by the attribute-to-parameter network and by relationship-reliability analyses | Values are matched to the 531 retained basins and treated as time-invariant descriptors. |
| Basin filtering and matching | 531 CAMELS-US basins with complete learned-parameter outputs, matched static attributes, and matched coordinates | Defines the common sample for model comparison, relationship analysis, and mapping | Runs are filtered to complete 531-basin outputs and the 14 analyzed HBV/routing parameters before relationship diagnostics. |
| Temporal split | Training: 1 January 1989 to 31 December 1998; testing: 1 January 1999 to 31 December 2009 | Separates model fitting from independent streamflow evaluation | All runs use the same dates. A 365-day warm-up is removed before comparing simulated and observed streamflow. |
| Neural-network input preprocessing | Static basin-attribute tensor supplied to the parameterization network | Provides normalized static inputs for learning basin-specific parameter vectors | The learning pipeline uses normalized static inputs internally for neural-network training and inference; relationship analyses use the matched basin-attribute table on the reported CAMELS attribute scale. |

**Table note.** Dynamic forcings drive the hydrologic simulation, whereas static attributes condition the learned parameter vector. Attribute-parameter analyses are cross-basin analyses because the attributes are time-invariant basin descriptors. The table reports the variables used in the 531-basin analysis and summarizes preprocessing relevant to model fitting, streamflow evaluation, and relationship diagnostics.

For deterministic runs, each basin has one parameter estimate per run. For dropout and distributional runs, stochastic outputs provide parameter samples or parameter-scale spread. Dropout and distributional uncertainty are not identical uncertainty concepts; therefore uncertainty comparisons are interpreted diagnostically.

## Appendix C. Relationship-reliability workflow

The relationship-reliability workflow has four stages. First, parameter estimates are processed into one run-level basin-parameter table. Second, seed-wise attribute-parameter Spearman matrices are computed for each model, loss, and seed. Third, cross-seed and cross-loss variability are summarized by standard deviation, range, sign consistency, dominant-attribute consistency, and top-k overlap. Fourth, model-consistency classes are assigned by comparing modal dominant attributes and signs across formulations.

Candidate relationships used in the main seed-stability figure are selected by mean absolute Spearman rho across all formulations, losses, and seeds. This prevents cherry-picking relationships favorable to \(\delta_{\mathrm{dist}}\). Full-pair summaries are retained in the Supporting Information because many weak pairs have little hydrologic meaning.

## Appendix D. Boundary and uncertainty diagnostics

Boundary diagnostics are needed because several parameters are frequently near lower or upper search-range limits. Boundary-sensitive parameters include \(\mathrm{BETA}\), \(\mathrm{CFMAX}\), \(\mathrm{CFR}\), \(\mathrm{CWH}\), \(\mathrm{K}_0\), \(\mathrm{K}_1\), \(\mathrm{K}_2\), \(\mathrm{LP}\), \(\mathrm{UH}_a\), and \(\mathrm{UH}_b\). Apparent stability near a boundary can reflect transformation limits rather than reliable learning.

For uncertainty interpretation, mean-standard deviation coupling measures whether parameter standard deviation follows the parameter mean, and boundary distance-standard deviation coupling measures whether uncertainty is controlled by proximity to a parameter bound. Parameters with high coupling or boundary sensitivity are flagged as requiring cautious interpretation. These flags constrain uncertainty claims to diagnostic structure rather than physical identifiability.

## Appendix E. Supporting Information figure guide

**Figure S1. Full predictive-performance summaries.** Extended NSE, KGE, and bias distributions by model and loss.

**Figure S2. Boundary-threshold sensitivity.** Sensitivity of boundary-saturation diagnostics to alternative boundary thresholds.

**Figure S3. Top-k sensitivity.** Relationship-stability summaries under alternative top-k candidate relationship definitions.

**Figure S4. Performance-filtered sensitivity.** Relationship-stability results after applying predictive-performance filters.

**Figure S5. Regional stability comparison.** Additional group-wise relationship-stability summaries.

**Figure S6. Spatial maps for deterministic and dropout formulations.** Baseline and dropout maps corresponding to the main distributional maps.

**Figure S7. Mean attribute-parameter matrices for baseline and dropout formulations.** Full relationship heatmaps for non-distributional formulations.

**Figure S8. Monte Carlo dropout uncertainty structure.** Dropout parameter-spread diagnostics.

**Figure S9. Uncertainty-width comparison.** Comparison of stochastic interval widths for dropout and distributional formulations.

**Figure S10. Explainability checks.** Post-hoc feature-importance summaries for selected parameters.

**Figure S11. Joint-effect heatmaps.** Two-attribute views for selected parameter gradients.

**Figure S12. Extended mean and uncertainty gradients.** Additional distributional mean and standard deviation gradients.

# Appendix F Supplementary Tables

These tables were filled using the extended reliability diagnostics in `manuscript/extends/`.

---

## Table S3. Parameter-value stability versus relationship-stability mismatch diagnostics.

| Formulation | Parameter | Parameter group | Raw parameter cross-seed SD | Boundary saturation fraction | Relationship cross-seed SD of Spearman rho | Dominant-control consistency | Mismatch category                       | Diagnostic note                                                                                                 |
| ----------- | --------- | --------------- | ---------------------------:| ----------------------------:| ------------------------------------------:| ----------------------------:| --------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| \(\delta_{\mathrm{base}}\)  | \(\mathrm{CFR}\)   | Snow/canopy     | 0.0000                      | 1.000                        | 0.119                                      | 0.333                        | Stable value / less stable relationship | Low cross-seed parameter spread coincides with near-boundary estimates and weaker dominant-control consistency. |
| \(\delta_{\mathrm{base}}\)  | \(\mathrm{CWH}\)   | Snow/canopy     | 0.0000                      | 1.000                        | 0.197                                      | 0.467                        | Stable value / less stable relationship | Low cross-seed parameter spread coincides with near-boundary estimates and weaker dominant-control consistency. |
| \(\delta_{\mathrm{base}}\)  | \(\mathrm{K}_2\)   | Recession       | 0.0144                      | 0.277                        | 0.018                                      | 0.600                        | Low variability / boundary saturation   | Low cross-seed parameter spread coincides with elevated near-boundary estimates.                                |
| \(\delta_{\mathrm{base}}\)  | \(\mathrm{UH}_a\)  | Routing         | 0.0177                      | 0.668                        | 0.120                                      | 0.667                        | Stable value / less stable relationship | Low cross-seed parameter spread coincides with near-boundary estimates and weaker dominant-control consistency. |
| \(\delta_{\mathrm{dist}}\)  | \(\mathrm{K}_2\)   | Recession       | 0.0150                      | 0.266                        | 0.015                                      | 0.533                        | Low variability / boundary saturation   | Low cross-seed parameter spread coincides with elevated near-boundary estimates.                                |
| \(\delta_{\mathrm{mcd}}\)   | \(\mathrm{CFMAX}\) | Snow/canopy     | 0.0148                      | 0.096                        | 0.049                                      | 0.333                        | Stable value / less stable relationship | Apparent value stability does not imply stable dominant controls.                                               |
| \(\delta_{\mathrm{mcd}}\)   | \(\mathrm{CFR}\)   | Snow/canopy     | 0.0000                      | 1.000                        | 0.062                                      | 0.600                        | Stable value / less stable relationship | Low cross-seed parameter spread coincides with near-boundary estimates and weaker dominant-control consistency. |
| \(\delta_{\mathrm{mcd}}\)   | \(\mathrm{CWH}\)   | Snow/canopy     | 0.0000                      | 1.000                        | 0.057                                      | 0.467                        | Stable value / less stable relationship | Low cross-seed parameter spread coincides with near-boundary estimates and weaker dominant-control consistency. |
| \(\delta_{\mathrm{mcd}}\)   | \(\mathrm{K}_2\)   | Recession       | 0.0167                      | 0.235                        | 0.017                                      | 0.467                        | Stable value / less stable relationship | Apparent value stability does not imply stable dominant controls.                                               |

**Table note.** Raw parameter stability and relationship stability were evaluated separately. Boundary saturation fraction refers to the proportion of basin-level normalized parameter estimates located near the prescribed parameter bounds. Mismatch categories identify cases where low apparent parameter-value variability did not correspond to stable attribute-parameter relationships.

---

## Table S4. Leave-one-group-out and within-group sensitivity for selected distributional relationships.

| Relationship          | Response type | All-basin Spearman rho | Leave-out group | Spearman rho after exclusion | Delta rho | Max absolute Delta rho across groups | Within-group rho range | Sign change within groups? | Diagnostic note                                                                                                  |
| --------------------- | ------------- | ----------------------:| --------------- | ----------------------------:| ---------:| ------------------------------------:| ---------------------- | -------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| \(\mathrm{BETA}\)-slope_mean       | Mean          | -0.587                 | G1              | -0.615                       | -0.028    | 0.049                                | -0.731 to -0.409       | No                         | Leave-out estimates are stable; within-group signs are retained.                                                 |
| \(\mathrm{BETA}\)-slope_mean       | Mean          | -0.587                 | G2              | -0.627                       | -0.040    | 0.049                                | -0.731 to -0.409       | No                         | Leave-out estimates are stable; within-group signs are retained.                                                 |
| \(\mathrm{BETA}\)-slope_mean       | Mean          | -0.587                 | G3              | -0.606                       | -0.019    | 0.049                                | -0.731 to -0.409       | No                         | Leave-out estimates are stable; within-group signs are retained.                                                 |
| \(\mathrm{BETA}\)-slope_mean       | Mean          | -0.587                 | G4              | -0.552                       | 0.035     | 0.049                                | -0.731 to -0.409       | No                         | Leave-out estimates are stable; within-group signs are retained.                                                 |
| \(\mathrm{BETA}\)-slope_mean       | Mean          | -0.587                 | G5              | -0.561                       | 0.026     | 0.049                                | -0.731 to -0.409       | No                         | Leave-out estimates are stable; within-group signs are retained.                                                 |
| \(\mathrm{BETA}\)-slope_mean       | Mean          | -0.587                 | G6              | -0.611                       | -0.024    | 0.049                                | -0.731 to -0.409       | No                         | Leave-out estimates are stable; within-group signs are retained.                                                 |
| \(\mathrm{BETA}\)-slope_mean       | Mean          | -0.587                 | G7              | -0.538                       | 0.049     | 0.049                                | -0.731 to -0.409       | No                         | Leave-out estimates are stable; within-group signs are retained.                                                 |
| \(\mathrm{FC}\)-pet_mean           | Mean          | 0.540                  | G1              | 0.482                        | -0.058    | 0.058                                | -0.096 to 0.615        | Yes                        | Leave-out estimates remain same-signed, but at least one within-group estimate changes sign or weakens strongly. |
| \(\mathrm{FC}\)-pet_mean           | Mean          | 0.540                  | G2              | 0.521                        | -0.019    | 0.058                                | -0.096 to 0.615        | Yes                        | Leave-out estimates remain same-signed, but at least one within-group estimate changes sign or weakens strongly. |
| \(\mathrm{FC}\)-pet_mean           | Mean          | 0.540                  | G3              | 0.492                        | -0.048    | 0.058                                | -0.096 to 0.615        | Yes                        | Leave-out estimates remain same-signed, but at least one within-group estimate changes sign or weakens strongly. |
| \(\mathrm{FC}\)-pet_mean           | Mean          | 0.540                  | G4              | 0.561                        | 0.021     | 0.058                                | -0.096 to 0.615        | Yes                        | Leave-out estimates remain same-signed, but at least one within-group estimate changes sign or weakens strongly. |
| \(\mathrm{FC}\)-pet_mean           | Mean          | 0.540                  | G5              | 0.539                        | -0.001    | 0.058                                | -0.096 to 0.615        | Yes                        | Leave-out estimates remain same-signed, but at least one within-group estimate changes sign or weakens strongly. |
| \(\mathrm{FC}\)-pet_mean           | Mean          | 0.540                  | G6              | 0.584                        | 0.044     | 0.058                                | -0.096 to 0.615        | Yes                        | Leave-out estimates remain same-signed, but at least one within-group estimate changes sign or weakens strongly. |
| \(\mathrm{FC}\)-pet_mean           | Mean          | 0.540                  | G7              | 0.546                        | 0.006     | 0.058                                | -0.096 to 0.615        | Yes                        | Leave-out estimates remain same-signed, but at least one within-group estimate changes sign or weakens strongly. |
| \(\mathrm{PERC}\)-aridity          | Mean          | -0.390                 | G1              | -0.291                       | 0.099     | 0.114                                | -0.525 to 0.088        | Yes                        | Leave-out estimates remain same-signed, but at least one within-group estimate changes sign or weakens strongly. |
| \(\mathrm{PERC}\)-aridity          | Mean          | -0.390                 | G2              | -0.369                       | 0.021     | 0.114                                | -0.525 to 0.088        | Yes                        | Leave-out estimates remain same-signed, but at least one within-group estimate changes sign or weakens strongly. |
| \(\mathrm{PERC}\)-aridity          | Mean          | -0.390                 | G3              | -0.437                       | -0.048    | 0.114                                | -0.525 to 0.088        | Yes                        | Leave-out estimates remain same-signed, but at least one within-group estimate changes sign or weakens strongly. |
| \(\mathrm{PERC}\)-aridity          | Mean          | -0.390                 | G4              | -0.384                       | 0.006     | 0.114                                | -0.525 to 0.088        | Yes                        | Leave-out estimates remain same-signed, but at least one within-group estimate changes sign or weakens strongly. |
| \(\mathrm{PERC}\)-aridity          | Mean          | -0.390                 | G5              | -0.423                       | -0.034    | 0.114                                | -0.525 to 0.088        | Yes                        | Leave-out estimates remain same-signed, but at least one within-group estimate changes sign or weakens strongly. |
| \(\mathrm{PERC}\)-aridity          | Mean          | -0.390                 | G6              | -0.503                       | -0.114    | 0.114                                | -0.525 to 0.088        | Yes                        | Leave-out estimates remain same-signed, but at least one within-group estimate changes sign or weakens strongly. |
| \(\mathrm{PERC}\)-aridity          | Mean          | -0.390                 | G7              | -0.302                       | 0.088     | 0.114                                | -0.525 to 0.088        | Yes                        | Leave-out estimates remain same-signed, but at least one within-group estimate changes sign or weakens strongly. |
| \(\mathrm{UZL}\)-soil_conductivity | Mean          | 0.609                  | G1              | 0.652                        | 0.043     | 0.086                                | 0.213 to 0.768         | No                         | Leave-out estimates are same-signed, with moderate within-group magnitude variation.                             |
| \(\mathrm{UZL}\)-soil_conductivity | Mean          | 0.609                  | G2              | 0.631                        | 0.022     | 0.086                                | 0.213 to 0.768         | No                         | Leave-out estimates are same-signed, with moderate within-group magnitude variation.                             |
| \(\mathrm{UZL}\)-soil_conductivity | Mean          | 0.609                  | G3              | 0.522                        | -0.086    | 0.086                                | 0.213 to 0.768         | No                         | Leave-out estimates are same-signed, with moderate within-group magnitude variation.                             |
| \(\mathrm{UZL}\)-soil_conductivity | Mean          | 0.609                  | G4              | 0.561                        | -0.048    | 0.086                                | 0.213 to 0.768         | No                         | Leave-out estimates are same-signed, with moderate within-group magnitude variation.                             |
| \(\mathrm{UZL}\)-soil_conductivity | Mean          | 0.609                  | G5              | 0.639                        | 0.030     | 0.086                                | 0.213 to 0.768         | No                         | Leave-out estimates are same-signed, with moderate within-group magnitude variation.                             |
| \(\mathrm{UZL}\)-soil_conductivity | Mean          | 0.609                  | G6              | 0.594                        | -0.014    | 0.086                                | 0.213 to 0.768         | No                         | Leave-out estimates are same-signed, with moderate within-group magnitude variation.                             |
| \(\mathrm{UZL}\)-soil_conductivity | Mean          | 0.609                  | G7              | 0.644                        | 0.036     | 0.086                                | 0.213 to 0.768         | No                         | Leave-out estimates are same-signed, with moderate within-group magnitude variation.                             |
| \(\mathrm{CWH}\)-frac_snow         | Mean          | -0.847                 | G1              | -0.851                       | -0.004    | 0.042                                | -0.915 to -0.604       | No                         | Leave-out estimates are stable; within-group signs are retained.                                                 |
| \(\mathrm{CWH}\)-frac_snow         | Mean          | -0.847                 | G2              | -0.805                       | 0.042     | 0.042                                | -0.915 to -0.604       | No                         | Leave-out estimates are stable; within-group signs are retained.                                                 |
| \(\mathrm{CWH}\)-frac_snow         | Mean          | -0.847                 | G3              | -0.847                       | 0.000     | 0.042                                | -0.915 to -0.604       | No                         | Leave-out estimates are stable; within-group signs are retained.                                                 |
| \(\mathrm{CWH}\)-frac_snow         | Mean          | -0.847                 | G4              | -0.882                       | -0.035    | 0.042                                | -0.915 to -0.604       | No                         | Leave-out estimates are stable; within-group signs are retained.                                                 |
| \(\mathrm{CWH}\)-frac_snow         | Mean          | -0.847                 | G5              | -0.814                       | 0.034     | 0.042                                | -0.915 to -0.604       | No                         | Leave-out estimates are stable; within-group signs are retained.                                                 |
| \(\mathrm{CWH}\)-frac_snow         | Mean          | -0.847                 | G6              | -0.867                       | -0.020    | 0.042                                | -0.915 to -0.604       | No                         | Leave-out estimates are stable; within-group signs are retained.                                                 |
| \(\mathrm{CWH}\)-frac_snow         | Mean          | -0.847                 | G7              | -0.845                       | 0.002     | 0.042                                | -0.915 to -0.604       | No                         | Leave-out estimates are stable; within-group signs are retained.                                                 |
| \(\mathrm{CFR}\)-frac_snow         | Mean          | -0.697                 | G1              | -0.728                       | -0.031    | 0.038                                | -0.860 to -0.274       | No                         | Leave-out estimates are stable; within-group signs are retained.                                                 |
| \(\mathrm{CFR}\)-frac_snow         | Mean          | -0.697                 | G2              | -0.659                       | 0.038     | 0.038                                | -0.860 to -0.274       | No                         | Leave-out estimates are stable; within-group signs are retained.                                                 |
| \(\mathrm{CFR}\)-frac_snow         | Mean          | -0.697                 | G3              | -0.717                       | -0.020    | 0.038                                | -0.860 to -0.274       | No                         | Leave-out estimates are stable; within-group signs are retained.                                                 |
| \(\mathrm{CFR}\)-frac_snow         | Mean          | -0.697                 | G4              | -0.689                       | 0.008     | 0.038                                | -0.860 to -0.274       | No                         | Leave-out estimates are stable; within-group signs are retained.                                                 |
| \(\mathrm{CFR}\)-frac_snow         | Mean          | -0.697                 | G5              | -0.694                       | 0.003     | 0.038                                | -0.860 to -0.274       | No                         | Leave-out estimates are stable; within-group signs are retained.                                                 |
| \(\mathrm{CFR}\)-frac_snow         | Mean          | -0.697                 | G6              | -0.683                       | 0.014     | 0.038                                | -0.860 to -0.274       | No                         | Leave-out estimates are stable; within-group signs are retained.                                                 |
| \(\mathrm{CFR}\)-frac_snow         | Mean          | -0.697                 | G7              | -0.675                       | 0.022     | 0.038                                | -0.860 to -0.274       | No                         | Leave-out estimates are stable; within-group signs are retained.                                                 |

**Table note.** Leave-one-group-out sensitivity was computed by removing one hydroclimatic stratum at a time and recalculating the Spearman relationship in the remaining basins. Within-group correlations were computed separately within each hydroclimatic stratum when sample size was sufficient. These summaries are sensitivity diagnostics and are not ungauged-basin transfer tests.

---

## Table S5. Strong collinearity among selected CAMELS basin descriptors.

| Attribute 1    | Attribute 2    | Spearman rho | Absolute rho | Attribute group 1 | Attribute group 2 | Threshold class       | Relevance to interpretation                                     |
| -------------- | -------------- | ------------:| ------------:| ----------------- | ----------------- | --------------------- | --------------------------------------------------------------- |
| high_prec_freq | low_prec_freq  | 0.939        | 0.939        | Climate           | Climate           | abs(rho) >= 0.8       | Relevant to precipitation-frequency gradient interpretation     |
| aridity        | p_mean         | -0.867       | 0.867        | Climate           | Climate           | abs(rho) >= 0.8       | Relevant to \(\mathrm{PERC}\)-aridity and \(\mathrm{FC}\)-PET interpretation              |
| slope_mean     | elev_mean      | 0.807        | 0.807        | Topography        | Topography        | abs(rho) >= 0.8       | Relevant to \(\mathrm{BETA}\)-slope and snow-related gradient interpretation |
| slope_mean     | soil_depth     | -0.798       | 0.798        | Topography        | Soil              | 0.6 <= abs(rho) < 0.8 | Relevant to \(\mathrm{UZL}\)-soil and \(\mathrm{BETA}\)-slope interpretation              |
| aridity        | low_prec_freq  | 0.789        | 0.789        | Climate           | Climate           | 0.6 <= abs(rho) < 0.8 | Relevant to \(\mathrm{PERC}\)-aridity interpretation                         |
| aridity        | forest_frac    | -0.706       | 0.706        | Climate           | Vegetation        | 0.6 <= abs(rho) < 0.8 | Relevant to climate-vegetation gradient interpretation          |
| forest_frac    | low_prec_freq  | -0.701       | 0.701        | Vegetation        | Climate           | 0.6 <= abs(rho) < 0.8 | Relevant to climate-vegetation gradient interpretation          |
| elev_mean      | frac_snow      | 0.687        | 0.687        | Topography        | Climate/snow      | 0.6 <= abs(rho) < 0.8 | Relevant to snow-related gradients                              |
| lai_diff       | high_prec_dur  | -0.678       | 0.678        | Vegetation        | Climate           | 0.6 <= abs(rho) < 0.8 | Relevant to climate-vegetation gradient interpretation          |
| elev_mean      | soil_depth     | -0.664       | 0.664        | Topography        | Soil              | 0.6 <= abs(rho) < 0.8 | Relevant to topography-soil gradient interpretation             |
| aridity        | high_prec_freq | 0.661        | 0.661        | Climate           | Climate           | 0.6 <= abs(rho) < 0.8 | Relevant to \(\mathrm{PERC}\)-aridity interpretation                         |
| aridity        | lai_diff       | -0.651       | 0.651        | Climate           | Vegetation        | 0.6 <= abs(rho) < 0.8 | Relevant to climate-vegetation gradient interpretation          |
| forest_frac    | lai_diff       | 0.638        | 0.638        | Vegetation        | Vegetation        | 0.6 <= abs(rho) < 0.8 | Relevant to vegetation-gradient interpretation                  |
| forest_frac    | high_prec_freq | -0.630       | 0.630        | Vegetation        | Climate           | 0.6 <= abs(rho) < 0.8 | Relevant to climate-vegetation gradient interpretation          |
| p_mean         | forest_frac    | 0.629        | 0.629        | Climate           | Vegetation        | 0.6 <= abs(rho) < 0.8 | Relevant to climate-vegetation gradient interpretation          |
| p_mean         | low_prec_freq  | -0.613       | 0.613        | Climate           | Climate           | 0.6 <= abs(rho) < 0.8 | Relevant to precipitation-frequency gradient interpretation     |
| slope_mean     | p_seasonality  | -0.602       | 0.602        | Topography        | Climate           | 0.6 <= abs(rho) < 0.8 | Relevant to slope-related interpretation                        |
| aridity        | pet_mean       | 0.600        | 0.600        | Climate           | Climate           | 0.6 <= abs(rho) < 0.8 | Relevant to \(\mathrm{FC}\)-PET and \(\mathrm{PERC}\)-aridity interpretation              |

**Table note.** Spearman correlations were computed among selected basin attributes used in the interpretation of learned parameter gradients. These correlations are used to document attribute collinearity and to support interpreting dominant attributes as representatives of broader hydroclimatic or physiographic gradients, rather than isolated causal controls.

---

## Table S6. Residual sensitivity of the four main shared behavioral gradients.

| Parameter | Target attribute  | Control attributes                  | Original Spearman rho | Residual Spearman rho | Direction retained? | Absolute change in rho | Evidence interpretation                                                        |
| --------- | ----------------- | ----------------------------------- | ---------------------:| ---------------------:| ------------------- | ----------------------:| ------------------------------------------------------------------------------ |
| \(\mathrm{BETA}\) | slope_mean        | elev_mean; frac_snow                | -0.523                | -0.335                | Yes                 | 0.188                  | Direction retained after residualizing; magnitude weakened                     |
| \(\mathrm{FC}\)   | pet_mean          | aridity; p_mean                     | 0.487                 | 0.281                 | Yes                 | 0.206                  | Direction retained after residualizing; magnitude weakened                     |
| \(\mathrm{PERC}\) | aridity           | pet_mean; p_mean                    | -0.481                | -0.221                | Yes                 | 0.260                  | Direction retained but more strongly affected by correlated climate attributes |
| \(\mathrm{UZL}\)  | soil_conductivity | soil_depth; slope_mean; forest_frac | 0.534                 | 0.460                 | Yes                 | 0.074                  | Direction and magnitude comparatively retained                                 |

**Table note.** Residual sensitivity was computed by rank-residualizing both the learned distributional parameter mean and the target basin attribute against selected correlated descriptors, then computing Spearman correlation between the residuals. Values may differ from reference-gradient summaries because this diagnostic uses an aggregated distributional relationship dataset. The analysis is a collinearity sensitivity check and does not isolate causal effects.

---

## Table S7. Distributional uncertainty diagnostic classification.

| Parameter std | Dominant attribute | Spearman rho | Mean-std coupling rho | Boundary-distance coupling rho | Near-boundary share | Diagnostic class                    | Evidence label                    | Diagnostic note                                        |
| ------------- | ------------------ | ------------:| ---------------------:| ------------------------------:| -------------------:| ----------------------------------- | --------------------------------- | ------------------------------------------------------ |
| \(\mathrm{BETA}\) std      | p_mean             | 0.2750       | 0.544                 | 0.818                          | 0.539               | Mean-coupled and boundary-sensitive | Cautionary uncertainty gradient   | Affected by mean-std coupling and boundary sensitivity |
| \(\mathrm{CFMAX}\) std     | frac_snow          | -0.9185      | 0.042                 | 0.039                          | 0.147               | Less-confounded                     | Structured uncertainty diagnostic | Low coupling and low boundary sensitivity              |
| \(\mathrm{CFR}\) std       | lai_diff           | 0.5891       | 0.451                 | 0.500                          | 0.000               | Boundary-sensitive                  | Cautionary uncertainty gradient   | Affected by boundary sensitivity                       |
| \(\mathrm{CWH}\) std       | frac_snow          | -0.9221      | 0.988                 | 0.991                          | 0.503               | Mean-coupled and boundary-sensitive | Cautionary uncertainty gradient   | Affected by mean-std coupling and boundary sensitivity |
| \(\mathrm{FC}\) std        | p_seasonality      | -0.3182      | 0.659                 | 0.800                          | 0.241               | Mean-coupled and boundary-sensitive | Cautionary uncertainty gradient   | Affected by mean-std coupling and boundary sensitivity |
| \(\mathrm{K}_0\) std        | aridity            | -0.2290      | -0.415                | 0.908                          | 0.738               | Boundary-sensitive                  | Cautionary uncertainty gradient   | Affected by boundary sensitivity                       |
| \(\mathrm{K}_1\) std        | sand_frac          | -0.4172      | 0.753                 | 0.157                          | 0.137               | Mean-coupled                        | Cautionary uncertainty gradient   | Affected by mean-std coupling                          |
| \(\mathrm{K}_2\) std        | elev_mean          | -0.4486      | 0.906                 | 0.893                          | 0.264               | Mean-coupled and boundary-sensitive | Cautionary uncertainty gradient   | Affected by mean-std coupling and boundary sensitivity |
| \(\mathrm{LP}\) std        | aridity            | -0.3402      | 0.132                 | 0.710                          | 0.384               | Boundary-sensitive                  | Cautionary uncertainty gradient   | Affected by boundary sensitivity                       |
| \(\mathrm{PERC}\) std      | aridity            | -0.5845      | 0.862                 | 0.883                          | 0.343               | Mean-coupled and boundary-sensitive | Cautionary uncertainty gradient   | Affected by mean-std coupling and boundary sensitivity |
| \(\mathrm{TT}\) std        | frac_snow          | -0.9105      | -0.181                | -0.176                         | 0.009               | Less-confounded                     | Structured uncertainty diagnostic | Low coupling and low boundary sensitivity              |
| \(\mathrm{UH}_a\) std      | lai_diff           | -0.5369      | 0.491                 | 0.681                          | 0.589               | Boundary-sensitive                  | Cautionary uncertainty gradient   | Affected by boundary sensitivity                       |
| \(\mathrm{UH}_b\) std      | frac_snow          | 0.5987       | 0.818                 | 0.896                          | 0.629               | Mean-coupled and boundary-sensitive | Cautionary uncertainty gradient   | Affected by mean-std coupling and boundary sensitivity |
| \(\mathrm{UZL}\) std       | soil_conductivity  | 0.5779       | 0.840                 | 0.790                          | 0.373               | Mean-coupled and boundary-sensitive | Cautionary uncertainty gradient   | Affected by mean-std coupling and boundary sensitivity |

**Table note.** Distributional standard deviations were evaluated as parameter-scale spread diagnostics, not as calibrated Bayesian posterior uncertainty. Mean-std coupling indicates association between parameter mean and parameter standard deviation. Boundary-distance coupling indicates association between parameter spread and distance to the parameter bounds. Near-boundary share summarizes the fraction of basins with parameter means close to the prescribed bounds. Rows flagged as mean-coupled or boundary-sensitive should be interpreted as structured diagnostic patterns rather than less-confounded uncertainty gradients.

---

## Optional main-text compact table. Evidence hierarchy for selected learned basin controls and uncertainty gradients.

| Relationship              | Response type | Spearman rho | Seed SD | Cross-loss SD | High-minus-low difference | Cross-formulation class | Diagnostic flag                   | Evidence level                                            |
| ------------------------- | ------------- | ------------:| -------:| -------------:| -------------------------:| ----------------------- | --------------------------------- | --------------------------------------------------------- |
| \(\mathrm{BETA}\)-slope_mean           | Mean          | -0.583       | 0.017   | 0.049         | -0.302                    | Shared                  | None / low                        | Reproducible behavioral gradient                          |
| \(\mathrm{FC}\)-pet_mean               | Mean          | 0.509        | 0.012   | 0.108         | 0.197                     | Shared                  | None / low                        | Reproducible but sensitivity-affected behavioral gradient |
| \(\mathrm{PERC}\)-aridity              | Mean          | -0.594       | 0.015   | 0.260         | -0.149                    | Shared                  | Collinearity sensitivity          | Reproducible but sensitivity-affected behavioral gradient |
| \(\mathrm{UZL}\)-soil_conductivity     | Mean          | 0.570        | 0.018   | 0.024         | 0.251                     | Shared                  | None / low                        | Reproducible behavioral gradient                          |
| \(\mathrm{CWH}\)-frac_snow             | Mean          | -0.904       | 0.012   | 0.138         | -0.342                    | Partially shared        | Formulation sensitivity           | Distributional diagnostic gradient                        |
| \(\mathrm{CFR}\)-frac_snow             | Mean          | -0.625       | 0.063   | 0.052         | -0.081                    | Partially shared        | Formulation sensitivity           | Distributional diagnostic gradient                        |
| \(\mathrm{CFMAX}\) std-frac_snow       | Std           | -0.919       | 0.011   | 0.016         | -0.046                    | NA                      | Less-confounded                   | Structured uncertainty diagnostic                         |
| \(\mathrm{TT}\) std-frac_snow          | Std           | -0.910       | 0.008   | 0.020         | -0.083                    | NA                      | Less-confounded                   | Structured uncertainty diagnostic                         |
| \(\mathrm{CWH}\) std-frac_snow         | Std           | -0.922       | 0.008   | 0.046         | -0.149                    | NA                      | Mean-coupled / boundary-sensitive | Cautionary uncertainty gradient                           |
| \(\mathrm{PERC}\) std-aridity          | Std           | -0.584       | 0.015   | 0.168         | -0.005                    | NA                      | Mean-coupled / boundary-sensitive | Cautionary uncertainty gradient                           |
| \(\mathrm{UZL}\) std-soil_conductivity | Std           | 0.578        | 0.033   | 0.019         | 0.009                     | NA                      | Mean-coupled / boundary-sensitive | Cautionary uncertainty gradient                           |
| \(\mathrm{UH}_b\) std-frac_snow        | Std           | 0.599        | 0.038   | 0.196         | 0.011                     | NA                      | Mean-coupled / boundary-sensitive | Cautionary uncertainty gradient                           |

**Table note.** This compact evidence hierarchy summarizes selected relationships discussed in the main text. It is intended to classify evidence strength rather than rank models by predictive performance. NA indicates that the cross-formulation class is not applicable to distributional standard-deviation rows.

# Acknowledgments

Acknowledgments and funding information will be completed before submission.

# Conflict of Interest Disclosure

The authors declare no conflicts of interest relevant to this manuscript.
