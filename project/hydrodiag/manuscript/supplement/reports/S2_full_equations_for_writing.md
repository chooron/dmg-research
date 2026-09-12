# S2 Formula Source Package

All equations below are the implemented full-model equations. The daily index is (t); (P_t,T_t,E_{p,t}) denote the input precipitation, temperature and PET tensors. `eps` is the model `nearzero` default (10^{-8}). All `clamp`, `min`, `max`, `where`, finite-kernel and normalization operations are retained.

## S2.1 Host model formulations

### S2.1.1 XAJ

#### Inputs and states

The full class accepts `precip`, `pet`, and `temp`; only (P_t) and (E_{p,t}) enter the Base XAJ daily kernel. The state at the beginning of day (t) is (WU_t,WL_t,WD_t,S_t,FR_t,QI_t,QG_t), plus the 14-sample surface-runoff buffer. (WU,WL,WD) are tension-water stores (mm), (S) is the free-water store (mm), (FR) is dimensionless, (QI,QG) are recursive interflow and groundwater output states, and the buffer stores surface-runoff depth. Full-model defaults are (WU_0=0.6UM, WL_0=0.6LM, WD_0=0.6DM, S_0=0.5SM, FR_0=QI_0=QG_0=0.1), and a zero buffer (models/xaj.py:467-501).

#### Daily flux equations in execution order

Define (WM=UM+LM+DM), (WMM=WM(1+B)), (MS=SM(1+EX)). First,

[
P'_t=\max(P_t,0),\qquad E^a_t=\max(kE_{p,t},0).
]

[
EU_t=\min(WU_t+P'_t,E^a_t).
]

Let
[
ED^{raw}_t=\begin{cases}C(E^a_t-EU_t)-WL_t,&WL_t<C,LM\ \text{and}\ WL_t<C(E^a_t-EU_t),\\0,&\text{otherwise},\end{cases}qquad ED_t=\min(ED^{raw}_t,WD_t).
]

[
EL_t=\begin{cases}0,&WU_t+P'_t\ge E^a_t,\\
(E^a_t-EU_t)WL_t/(LM+\varepsilon),&WU_t+P'_t<E^a_t\ \text{and}\ WL_t\ge C,LM,\\
\min(C(E^a_t-EU_t),WL_t),&\text{otherwise}.
\end{cases}
\quad E_t=EU_t+EL_t+ED_t.
]

Set (W_{0,t}=\min(WU_t+WL_t+WD_t,WM-\varepsilon)), (PE_t=\max(P'_t-E_t,0)), (b_t=\max(1-W_{0,t}/(WM+\varepsilon),10^{-6})), and (A_t=WMM[1-b_t^{1/(1+B)}]). Then

[
R^{cal}_t=\begin{cases}
PE_t-(WM-W_{0,t})+WM\left[1-\min\left(\frac{A_t+PE_t}{WMM+\varepsilon},1\right)\right]^{1+B},&PE_t>0,\ PE_t+A_t<WMM,\\
PE_t-(WM-W_{0,t}),&PE_t>0,\ PE_t+A_t\ge WMM,\\
0,&PE_t\le0,
\end{cases}\qquad R_t=\max(R^{cal}_t,0),\quad R^{IM}_t=\max(PE_t IM,0).
]

The tension stores are updated using old-day stores on the right-hand side:

[
WU^*_t=\begin{cases}\min(WU_t+PE_t-R_t,UM),&PE_t>0,\\\max(WU_t+P'_t-E_t,0),&PE_t\le0,\end{cases}
]

[
WD^*_t=\begin{cases}WU_t+WL_t+WD_t+PE_t-R_t-UM-LM,&PE_t>0\ \text{and}\ WU_t+WL_t+PE_t-R_t>UM+LM,\\WD_t,&PE_t>0\ \text{and the condition is false},\\WD_t-ED_t,&PE_t\le0,\end{cases}
]

where the first branch uses the explicit capacity remainder shown above; and

[
WL^*_t=\begin{cases}WU_t+WL_t+WD_t+PE_t-R_t-WU^*_t-WD^*_t,&PE_t>0,\\WL_t-EL_t,&PE_t\le0.\end{cases}
\]

Finally (WU_{t+1}=\operatorname{clamp}(WU^*_t,0,UM)), (WL_{t+1}=\operatorname{clamp}(WL^*_t,0,LM)), and (WD_{t+1}=\operatorname{clamp}(WD^*_t,0,DM)).

For free-water separation, (m_t=1[R_t>0]), (FR^*_t=m_tR_t/(PE_t+\varepsilon)+(1-m_t)FR_t), and (FR_{t+1}=\operatorname{clamp}(FR^*_t,0,1)). Define (SS_t=m_t FR_tS_t/(FR^*_t+\varepsilon)+(1-m_t)S_t), then (SS_t=\min(SS_t,SM-\varepsilon)), (c^f_t=\max(1-SS_t/(SM+\varepsilon),10^{-6})), (AU_t=MS[1-(c^f_t)^{1/(1+EX)}]). Then

[
RS^{fr}_t=\begin{cases}FR_{t+1}\left[PE_t-SM+SS_t+SM\left(1-\min\left(\frac{PE_t+AU_t}{MS+\varepsilon},1\right)\right)^{1+EX}\right],&PE_t+AU_t<MS,\\FR_{t+1}(PE_t+SS_t-SM),&PE_t+AU_t\ge MS,
\end{cases}
]

[RS_t=\operatorname{clamp}(m_t\min(RS^{fr}_t,R_t),0,\infty),quad S^*_t=SS_t+m_t\frac{R_t-RS_t}{FR^*_t+\varepsilon},quad S_{t+1}=\min(S^*_t,SM-\varepsilon).]

[RI_t=KI S_{t+1}FR_{t+1},quad RG_t=KG S_{t+1}FR_{t+1}.]

The prepared XAJ parameters rescale (KI,KG) only when (KI+KG\ge1): they are multiplied by ((1-10^{-5})/\max(KI+KG,10^{-6})); otherwise they are unchanged (models/xaj.py:280-294). The recursive linear reservoirs and surface adjustment are

[QI_{t+1}=CI QI_t+(1-CI)RI_t(1-IM),quad QG_{t+1}=CG QG_t+(1-CG)RG_t(1-IM),quad RS^{adj}_t=RS_t(1-IM)+R^{IM}_t.]

The full path forms a 15-ordinate Gamma UH through `uh_gamma`, applies causal `uh_conv` to the 14-sample buffer concatenated with (RS^{adj}), and returns the current slice plus (QI_{t+1}+QG_{t+1}). The finite future tail is held in the continuation buffer and is not returned in the current finite window (models/xaj.py:297-342, 644-674).

#### Parameters

The active XAJ source specification contains 15 parameters: (k,B,IM,UM,LM,DM,C,SM,EX,KI,KG,CI,CG,a_{UH},\theta_{UH}), named in `models/parameter_specs.py:163-299`. Therefore any manuscript statement that the current active XAJ implementation has 14 parameters conflicts with the active source specification. Their exact code names, bounds and units are in `results/s2_parameter_inventory.csv`.

### S2.1.2 GR4J

States are (S^{prod}_t,S^{route}_t), UH1 buffer length 15 and UH2 buffer length 30. Defaults are (0.5X_1,0.5X_3), and zero buffers (models/gr4j.py:169-190).

[
M_t=1[P_t\ge E_{p,t}],\quad P^N_t=\max(P_t-E_{p,t},0),\quad E^N_t=\max(E_{p,t}-P_t,0),\quad r_t=\operatorname{clamp}(S^{prod}_t/(X_1+\varepsilon),0,1).
]

[
P^S_t=M_t\frac{X_1(1-r_t^2)\tanh(P^N_t/(X_1+\varepsilon))}{1+r_t\tanh(P^N_t/(X_1+\varepsilon))+\varepsilon},
]

[
E^S_t=(1-M_t)\frac{S^{prod}_t(2-r_t)\tanh(E^N_t/(X_1+\varepsilon))}{1+(1-r_t)\tanh(E^N_t/(X_1+\varepsilon))+\varepsilon}.
]

[S^{prod*}_t=S^{prod}_t-E^S_t+P^S_t,quad n_t=\frac49\frac{S^{prod*}_t}{X_1+\varepsilon},quad Perc_t=S^{prod*}_t[1-(1+n_t^4)^{-1/4}],]
[S^{prod}_{t+1}=S^{prod*}_t-Perc_t,quad P^R_t=\max(Perc_t+P^N_t-P^S_t,0),quad P^{R1}_t=.9P^R_t,quad P^{R2}_t=.1P^R_t.]

For (i=1,ldots,L), (L_1=15,L_2=30), (d=\max(X_4,10^{-3})), (u_i=i/d). UH1 uses (S_1(u)=\min(u,1)^{2.5}). UH2 uses (S_2(u)=0.5u^{2.5}) for (u\le1), and (1-0.5\max(2-u,0)^{2.5}) otherwise, followed by (min(S_2,1)). Ordinates are (w_i=S(i)-S(i-1)), divided by (sum_iw_i+10^{-8}) (models/unit_hydro.py:18-71).

Each UH buffer shifts left, appends zero, and adds (w_iP^{Rj}_t). Let (UH1_t,UH2_t) be the first entries. (F_t=X_2[\max(S^{route}_t/(X_3+\varepsilon),0)]^{3.5}). Then (S^{route*}_t=\max(S^{route}_t+UH1_t+F_t,0)), (Q^R_t=S^{route*}_t[1-(1+\max(S^{route*}_t/(X_3+\varepsilon),0)^4)^{-1/4}]), (S^{route}_{t+1}=S^{route*}_t-Q^R_t), (Q^D_t=\max(UH2_t+F_t,0)), and (Q_t=Q^R_t+Q^D_t). Buffers preserve finite-window continuation water (models/gr4j.py:63-91).

The active parameters (X_1,X_2,X_3,X_4) are `x1`--`x4`, with bounds and units in `models/parameter_specs.py:124-160`. Compared with canonical GR4J, the local implementation explicitly uses differentiable tensor S-curves, epsilon terms, finite buffers and local clamps; the formula above is the implemented formulation.

### S2.1.3 SIMHYD

The differentiable SIMHYD implementation used in this study has states (soil_t,groundwater_t) and a 14-sample UH buffer. Its daily equations are exactly the equations in the SIMHYD rows of `results/s2_formula_inventory.csv`: interception, exponential infiltration, direct runoff, interflow, recharge, soil ET, overflow transfer, groundwater recession and instantaneous runoff, followed by the normalized finite gamma UH. Initial states and routing are in `results/s2_initialization_inventory.csv` and `results/s2_routing_inventory.csv`. Parameters are the nine `simhyd_*` entries at `models/parameter_specs.py:432-473`.

### S2.2 Implemented CemaNeige module

The active module is the basic two-parameter variant with (CTG,K_f). Its complete partition, (G,eTG,SCA), melt and effective-input equations are given in the CN row of `results/s2_formula_inventory.csv` and source lines `models/cemaneige.py:16-74`. It is not the hysteresis class. PET is passed unchanged. The wrapper performs this module update first and feeds (P^{eff}_t=Rain_t+Melt_t) to the host on the same day.

### S2.3 Temperature-conditioned generic delay

The active TGD has (alpha,tau,beta), one storage (S_t), frozen training temperature mean and standard deviation, and equations in `models/temperature_delay.py:20-51`. The physical mapping is linear for (alpha,beta) and logarithmic for (tau) in `ablation/ic_core/parameter_adapter.py:56-67`; dPL first constrains network outputs with sigmoid at `training/dpl/run_dpl_model.py:166-168`. TGD uses temperature, conservatively redistributes precipitation through time, leaves PET unchanged, and contains no snow partition, SWE or melt equation.

### S2.4 Structural comparison of Base, TGD and CN

Base bypasses preprocessing; TGD adds one delay storage and three parameters; CN adds (G,eTG) and two parameters. Base/TGD/CN do not have equal state or parameter counts. TGD and CN both use (P,T) and leave PET unchanged, but only CN has explicit solid precipitation, snow storage and melt.

### S2.5 Coupling to host models

For each XAJ, GR4J and SIMHYD wrapper, the order is
[
(P_t,T_t,E_{p,t})\rightarrow\{P_t\ \text{(Base bypass)},\ P^{eff}_t\ \text{(TGD)},\ P^{eff}_t\ \text{(CN)}\}\rightarrow\text{host daily step}\rightarrow\text{host routing}\rightarrow Q_t.
]

The evidence is `models/composed.py:28-93`, `models/composed_temperature_delay.py:32-110`, and the host forward methods cited above. TGD/CN do not modify PET.

### S2.6 HBV snow-process reference

HBV is a standalone explicit snow-process reference. Its complete equations are the HBV rows in `results/s2_formula_inventory.csv`, source `models/hbv.py:30-71`; parameters and initial states are `models/parameter_specs.py:13-122` and `models/hbv.py:145-170`. It does not participate in Base/TGD/CN replacement wrappers.


## Expanded daily equations for SIMHYD, HBV, CN and TGD

### SIMHYD daily equations

Let INSC_s=max(INSC,1e-6), SMSC_s=max(SMSC,1e-6), COEFF_s=max(COEFF,1e-6), and rho_t=clamp(soil_t/(SMSC_s+eps),0,1). The exact daily order is:

$$P_s=max(P_t,0),\quad PET_s=max(ETMUL*PET_t,0),\quad I_t=min(INSC_s,PET_s,P_s),\quad T_t=P_s-I_t.$$

$$Icap_t=COEFF_s*exp(-SQ*rho_t),\quad Inf_t=min(Icap_t,T_t),\quad D_t=T_t-Inf_t.$$

$$IF_t=SUB*rho_t*Inf_t,\quad Rec_t=CRAK*rho_t*(Inf_t-IF_t),\quad SoilAvail_t=soil_t+Inf_t-IF_t-Rec_t.$$

$$ETsoil_t=min(10*rho_t,PET_s-I_t,SoilAvail_t),\quad SoilAfter_t=SoilAvail_t-ETsoil_t.$$

$$Overflow_t=max(SoilAfter_t-SMSC_s,0),\quad soil_{t+1}=SoilAfter_t-Overflow_t,$$
$$RecTotal_t=Rec_t+Overflow_t,\quad K_s=clamp(K,0,1),\quad BF_t=K_s*groundwater_t,$$
$$groundwater_{t+1}=groundwater_t+RecTotal_t-BF_t,\quad RunoffInst_t=D_t+IF_t+BF_t,\quad ET_t=I_t+ETsoil_t.$$

RunoffInst is concatenated with the 14-sample buffer, routed by the normalized length-15 gamma kernel, and sliced from index 14. The last 14 samples form the continuation buffer. Evidence: models/simhyd.py:52-118,165-200,224-297,329-353.

### HBV daily equations

$$RAIN_t=P_t*1[T_t>=TT],\quad SNOW_t=P_t*1[T_t<TT],\quad SNOWPACK^*_t=SNOWPACK_t+SNOW_t.$$
$$MeltPot_t=max(CFMAX*(T_t-TT),0),\quad Melt_t=min(MeltPot_t,SNOWPACK^*_t),\quad MELTWATER^*_t=MELTWATER_t+Melt_t.$$
$$RefreezePot_t=max(CFR*CFMAX*(TT-T_t),0),\quad Refreeze_t=min(RefreezePot_t,MELTWATER^*_t),$$
$$SNOWPACK^{**}_t=SNOWPACK^*_t+Refreeze_t,\quad MELTWATER^{**}_t=MELTWATER^*_t-Refreeze_t,$$
$$ToSoil_t=max(MELTWATER^{**}_t-CWH*SNOWPACK^{**}_t,0),\quad MELTWATER_{t+1}=MELTWATER^{**}_t-ToSoil_t.$$

$$w_t=clamp((SM_t/FC)^{BETA},0,1),\quad Recharge_t=(RAIN_t+ToSoil_t)w_t,$$
$$SM^*_t=SM_t+RAIN_t+ToSoil_t-Recharge_t,\quad Excess_t=max(SM^*_t-FC,0),\quad SM^{**}_t=SM^*_t-Excess_t,$$
$$ETfactor_t=clamp(SM^{**}_t/(LP*FC),0,1),\quad ETact_t=min(SM^{**}_t,PET_t*ETfactor_t),\quad SM_{t+1}=max(SM^{**}_t-ETact_t,nearzero).$$

$$SUZ^*_t=SUZ_t+Recharge_t+Excess_t,\quad Perc_t=min(SUZ^*_t,PERC),\quad SUZ^{**}_t=SUZ^*_t-Perc_t,$$
$$Q0_t=K0*max(SUZ^{**}_t-UZL,0),\quad SUZ^{***}_t=SUZ^{**}_t-Q0_t,\quad Q1_t=K1*SUZ^{***}_t,$$
$$SLZ^*_t=SLZ_t+Perc_t,\quad Q2_t=K2*SLZ^*_t,\quad Q_t=Q0_t+Q1_t+Q2_t.$$

Evidence: models/hbv.py:30-71,145-190. There is no convolutional routing kernel or continuation tail in HBV; Q0, Q1 and Q2 are same-day response outflows.

### Basic CemaNeige daily equations

$$fsolid_t=\begin{cases}1,&T_t<=0,\\0,&T_t>=3,\\1-(T_t+1)/4,&0<T_t<3,\end{cases}\quad Snow_t=P_t*fsolid_t,\quad Rain_t=P_t-Snow_t.$$
$$G^*_t=G_t+Snow_t,\quad eTG^*_t=CTG*eTG_t+(1-CTG)*T_t,\quad eTG_{t+1}=min(eTG^*_t,0).$$

With g_thresh=0.9*estimate_psol_annual(P,T), g_safe=max(g_thresh,nearzero),
$$SCA_t=0\quad\text{if }g_thresh<=nearzero,\qquad SCA_t=clamp(G^*_t/(g_safe+nearzero),0,1)\quad\text{otherwise}.$$
$$MeltPot_t=min(1[eTG_{t+1}=0 and T_t>0]*Kf*T_t,G^*_t),$$
$$Melt_t=min((0.9*SCA_t+0.1)*MeltPot_t,G^*_t),\quad G_{t+1}=G^*_t-Melt_t,\quad P^{eff}_t=Rain_t+Melt_t.$$

Evidence: models/cemaneige.py:16-74; the active fused wrappers call this basic step at models/composed.py:46-53.

### Temperature-conditioned generic delay equations

$$z_t=(T_t-Tmean_train)/max(Tstd_train,1e-6),\quad h_t=tanh(clamp(z_t,-5,5)),$$
$$tau_t=clamp(tau*exp(-beta*h_t),1e-6,3650),\quad f_t=-expm1(-1/tau_t).$$
$$Spre_t=S_t+alpha*P_t,\quad Release_t=f_t*Spre_t,\quad S_{t+1}=Spre_t-Release_t,$$
$$P^{eff}_t=(1-alpha)*P_t+Release_t.$$

The default TGD state is S_0=0. alpha and beta use linear physical bounds; tau uses logarithmic interpolation in ablation/ic_core/parameter_adapter.py:56-67. Evidence: models/temperature_delay.py:20-51,122-184 and models/composed_temperature_delay.py:32-110.
