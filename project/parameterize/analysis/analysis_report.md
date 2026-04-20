# Parameterization Analysis Report

## Accuracy summary by method

       variant  basin_count_mean  basin_count_std  kge_mean_empirical_mean  kge_mean_empirical_std  kge_median_empirical_mean  kge_median_empirical_std  kge_std_empirical_mean  kge_std_empirical_std  share_kge_gt_0_mean  share_kge_gt_0_std  share_kge_gt_05_mean  share_kge_gt_05_std  kge_mean_mean  kge_mean_std  kge_median_mean  kge_median_std  kge_std_mean  kge_std_std  nse_mean_mean  nse_mean_std  nse_median_mean  nse_median_std  nse_std_mean  nse_std_std  rmse_mean_mean  rmse_mean_std  rmse_median_mean  rmse_median_std  rmse_std_mean  rmse_std_std  corr_mean_mean  corr_mean_std  corr_median_mean  corr_median_std  corr_std_mean  corr_std_std  mae_mean_mean  mae_mean_std  mae_median_mean  mae_median_std  mae_std_mean  mae_std_std  pbias_abs_mean_mean  pbias_abs_mean_std  pbias_abs_median_mean  pbias_abs_median_std  pbias_abs_std_mean  pbias_abs_std_std
 deterministic             531.0              0.0                 0.601306                0.003724                   0.679345                  0.004713                0.366275               0.007418             0.964218            0.004983              0.799749             0.002877       0.601306      0.003724         0.679345        0.004713      0.366275     0.007418       0.456389      0.007844         0.632597        0.001827      2.235421     0.106867        1.561713       0.004656          1.330215         0.005147       1.066736      0.006270        0.770542       0.001431          0.812285         0.000061       0.145778      0.001143       0.631346      0.001165         0.505339        0.001916      0.510627     0.000766            51.409133            0.161748              45.845270              0.435353           20.631676           0.339110
distributional             531.0              0.0                 0.597989                0.001216                   0.677389                  0.003522                0.356169               0.015131             0.962963            0.001087              0.795982             0.005436       0.597989      0.001216         0.677389        0.003522      0.356169     0.015131       0.458674      0.008042         0.628319        0.003248      2.030368     0.288198        1.566124       0.002386          1.337444         0.004358       1.069022      0.003895        0.768450       0.001303          0.811666         0.000675       0.146376      0.002040       0.633726      0.000865         0.505355        0.003941      0.513019     0.001115            51.575673            0.084536              46.102481              0.245865           20.701600           0.013838
    mc_dropout             531.0              0.0                 0.594616                0.007550                   0.670749                  0.009691                0.312617               0.007507             0.960452            0.000000              0.792216             0.002877       0.594616      0.007550         0.670749        0.009691      0.312617     0.007507       0.494425      0.009651         0.642538        0.007595      1.308886     0.136105        1.544238       0.008582          1.317673         0.004734       1.060664      0.000801        0.774756       0.002072          0.819603         0.003520       0.143673      0.002595       0.622933      0.001434         0.494630        0.004339      0.500622     0.000337            51.182441            0.044470              45.823610              0.161931           20.564123           0.195830

## Top attribute-parameter correlations

       variant     target         attribute parameter  spearman_rho  abs_rho
 deterministic param_mean        slope_mean   parBETA     -0.661574 0.661574
 deterministic param_mean         frac_snow   route_b      0.641839 0.641839
 deterministic param_mean          lai_diff     parK1      0.594953 0.594953
 deterministic param_mean          pet_mean  parCFMAX     -0.593309 0.593309
 deterministic param_mean           aridity   parPERC     -0.587997 0.587997
    mc_dropout param_mean        slope_mean   parBETA     -0.643589 0.643589
    mc_dropout param_mean         frac_snow   route_b      0.637872 0.637872
    mc_dropout param_mean       frac_forest     parTT      0.632998 0.632998
    mc_dropout param_mean      low_prec_dur  parCFMAX     -0.614850 0.614850
    mc_dropout param_mean           aridity   parPERC     -0.606614 0.606614
    mc_dropout  param_std         frac_snow   route_b      0.576461 0.576461
    mc_dropout  param_std soil_conductivity    parUZL      0.561746 0.561746
    mc_dropout  param_std         elev_mean     parK2     -0.560313 0.560313
    mc_dropout  param_std        slope_mean   route_a     -0.556991 0.556991
    mc_dropout  param_std         frac_snow     parK2     -0.538019 0.538019
distributional param_mean         frac_snow    parCWH     -0.824499 0.824499
distributional param_mean         frac_snow    parCFR     -0.723836 0.723836
distributional param_mean        slope_mean   parBETA     -0.657297 0.657297
distributional param_mean          pet_mean    parCWH      0.657096 0.657096
distributional param_mean     low_prec_freq    parCWH      0.641807 0.641807
distributional  param_std         frac_snow     parTT     -0.854014 0.854014
distributional  param_std         frac_snow    parCWH     -0.835173 0.835173
distributional  param_std         frac_snow  parCFMAX     -0.830952 0.830952
distributional  param_std          pet_mean     parTT      0.690611 0.690611
distributional  param_std           aridity   route_a      0.683974 0.683974
