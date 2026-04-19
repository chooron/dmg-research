# Parameterization Analysis Report

## Accuracy summary by method

       variant  basin_count_mean  basin_count_std  kge_mean_empirical_mean  kge_mean_empirical_std  kge_median_empirical_mean  kge_median_empirical_std  kge_std_empirical_mean  kge_std_empirical_std  share_kge_gt_0_mean  share_kge_gt_0_std  share_kge_gt_05_mean  share_kge_gt_05_std  kge_mean_mean  kge_mean_std  kge_median_mean  kge_median_std  kge_std_mean  kge_std_std  nse_mean_mean  nse_mean_std  nse_median_mean  nse_median_std  nse_std_mean  nse_std_std  rmse_mean_mean  rmse_mean_std  rmse_median_mean  rmse_median_std  rmse_std_mean  rmse_std_std  corr_mean_mean  corr_mean_std  corr_median_mean  corr_median_std  corr_std_mean  corr_std_std  mae_mean_mean  mae_mean_std  mae_median_mean  mae_median_std  mae_std_mean  mae_std_std  pbias_abs_mean_mean  pbias_abs_mean_std  pbias_abs_median_mean  pbias_abs_median_std  pbias_abs_std_mean  pbias_abs_std_std
 deterministic             531.0              0.0                 0.601990                0.002077                   0.679149                  0.002730                0.366579               0.015956             0.964846            0.002877              0.801004             0.004739       0.601990      0.002077         0.679149        0.002730      0.366579     0.015956       0.456618      0.010380         0.628761        0.005584      2.243730     0.266692        1.561057       0.003141          1.329060         0.003861       1.063652      0.000452        0.770688       0.001766          0.812763         0.000665       0.144960      0.000742       0.631636      0.000930         0.503126        0.000821      0.511347     0.001057            51.378094            0.088652              45.938955              0.359792           20.399808           0.491990
distributional             531.0              0.0                 0.601909                0.004050                   0.679834                  0.003532                0.346003               0.017163             0.964846            0.003920              0.800377             0.003262       0.601909      0.004050         0.679834        0.003532      0.346003     0.017163       0.468732      0.017230         0.631496        0.002220      1.894446     0.274320        1.562014       0.002419          1.336120         0.004378       1.070568      0.006312        0.770666       0.001127          0.813064         0.000608       0.144687      0.002174       0.632583      0.001005         0.504654        0.000888      0.512468     0.001464            51.449911            0.035898              45.893246              0.190445           20.437815           0.179810
    mc_dropout             531.0              0.0                 0.576062                0.009580                   0.655892                  0.013374                0.312718               0.006500             0.956058            0.002175              0.760201             0.009479       0.576062      0.009580         0.655892        0.013374      0.312718     0.006500       0.474239      0.010004         0.632370        0.011282      1.122585     0.124688        1.560904       0.013943          1.326274         0.027326       1.062956      0.001213        0.769067       0.004127          0.817126         0.005236       0.146448      0.001224       0.630951      0.003420         0.505327        0.002431      0.501121     0.001526            52.244822            0.113160              46.942134              0.272061           21.066695           0.393937

## Top attribute-parameter correlations

       variant     target         attribute parameter  spearman_rho  abs_rho
 deterministic param_mean        slope_mean   parBETA     -0.656515 0.656515
 deterministic param_mean         frac_snow   route_b      0.648275 0.648275
 deterministic param_mean soil_conductivity    parUZL      0.594432 0.594432
 deterministic param_mean          pet_mean  parCFMAX     -0.591349 0.591349
 deterministic param_mean        slope_mean    parCWH     -0.584376 0.584376
    mc_dropout param_mean           aridity   parPERC     -0.704245 0.704245
    mc_dropout param_mean            p_mean   parPERC      0.680761 0.680761
    mc_dropout param_mean      low_prec_dur  parCFMAX     -0.654454 0.654454
    mc_dropout param_mean           gvf_max     parK0      0.653797 0.653797
    mc_dropout param_mean       frac_forest   parPERC      0.643122 0.643122
    mc_dropout  param_std         frac_snow   route_b      0.607978 0.607978
    mc_dropout  param_std     p_seasonality   parPERC     -0.601601 0.601601
    mc_dropout  param_std        slope_mean     parK2     -0.592391 0.592391
    mc_dropout  param_std           aridity     parK0      0.584890 0.584890
    mc_dropout  param_std       frac_forest     parK0     -0.578190 0.578190
distributional param_mean        slope_mean   parBETA     -0.663788 0.663788
distributional param_mean         frac_snow   route_b      0.621250 0.621250
distributional param_mean soil_conductivity    parUZL      0.591004 0.591004
distributional param_mean       frac_forest     parTT      0.586353 0.586353
distributional param_mean        slope_mean   parPERC      0.582398 0.582398
distributional  param_std         clay_frac  parCFMAX     -0.576908 0.576908
distributional  param_std          gvf_diff    parCFR      0.570618 0.570618
distributional  param_std     high_prec_dur    parCFR     -0.569817 0.569817
distributional  param_std       frac_forest     parTT      0.549229 0.549229
distributional  param_std           aridity     parK0      0.548838 0.548838
