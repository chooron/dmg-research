# Seed Stability Report

Focused parameter: `parFC`
Key parameters: `parFC`, `parBETA`, `parPERC`, `parUZL`, `parCFMAX`, `route_b`
Top-k overlap uses `k=5` strongest attributes by absolute Spearman rho.

## Overall parameter stability by method

       variant  parameter_count  mean_of_mean_basin_variance  median_of_mean_basin_variance  mean_of_mean_basin_std  mean_of_mean_normalized_basin_variance  median_of_mean_normalized_basin_variance
distributional               14                    69.234709                       0.040713                1.847554                                0.003910                                  0.003935
 deterministic               14                   119.643325                       0.046287                2.059094                                0.004218                                  0.002982
    mc_dropout               14                   113.668979                       0.028186                2.430166                                0.007513                                  0.003887

## Overall correlation R^2 stability by method

       variant  pair_count  mean_variance_spearman_r2  median_variance_spearman_r2  p90_variance_spearman_r2  max_variance_spearman_r2  mean_range_spearman_r2  mean_sign_consistency
distributional         490                   0.000142                     0.000039                  0.000343                  0.004961                0.019589               0.965986
 deterministic         490                   0.000802                     0.000089                  0.002070                  0.020641                0.040799               0.955782
    mc_dropout         490                   0.001269                     0.000175                  0.002731                  0.044748                0.051874               0.940816

## Sample-count validation

{
  "deterministic": 1,
  "distributional": 100,
  "mc_dropout": 100
}

## Basin variance summary for `parFC`

       variant parameter  basin_count  seed_count  observed_range  mean_basin_variance  median_basin_variance  mean_basin_std  median_basin_std  mean_basin_range  median_basin_range  mean_normalized_basin_variance  median_normalized_basin_variance
distributional     parFC          531           3      941.081646           940.367944             193.019806       21.291675         13.893157         49.619508           32.199566                        0.001062                          0.000218
    mc_dropout     parFC          531           3      936.021856          1557.369512             544.844449       28.942328         23.341903         67.387210           54.320732                        0.001778                          0.000622
 deterministic     parFC          531           3      945.306126          1650.091231             222.721059       24.439928         14.923842         56.777997           34.703445                        0.001847                          0.000249

## Lowest R^2 variance pairs for `parFC`

       variant            attribute parameter  seed_count  mean_spearman_rho  mean_abs_spearman_rho  variance_spearman_rho  std_spearman_rho  mean_spearman_r2  variance_spearman_r2  std_spearman_r2  min_spearman_r2  max_spearman_r2  sign_consistency  range_spearman_r2
 deterministic   soil_depth_statsgo     parFC           3          -0.002617               0.010330               0.000101          0.010031          0.000107          3.480057e-10         0.000019         0.000094         0.000134          0.666667           0.000040
 deterministic    max_water_content     parFC           3          -0.002425               0.009737               0.000114          0.010666          0.000120          8.011778e-09         0.000090         0.000010         0.000229          0.666667           0.000219
    mc_dropout   soil_depth_statsgo     parFC           3          -0.021164               0.021164               0.000011          0.003264          0.000459          1.766138e-08         0.000133         0.000279         0.000596          1.000000           0.000317
distributional  glim_1st_class_frac     parFC           3          -0.008321               0.010285               0.000100          0.009998          0.000169          4.129644e-08         0.000203         0.000009         0.000456          0.666667           0.000447
distributional  glim_2nd_class_frac     parFC           3           0.011518               0.011518               0.000083          0.009124          0.000216          6.194875e-08         0.000249         0.000004         0.000565          1.000000           0.000562
    mc_dropout    max_water_content     parFC           3          -0.038736               0.038736               0.000018          0.004206          0.001518          1.070569e-07         0.000327         0.001134         0.001934          1.000000           0.000800
distributional soil_depth_pelletier     parFC           3           0.015953               0.015953               0.000079          0.008873          0.000333          1.115118e-07         0.000334         0.000066         0.000804          1.000000           0.000738
distributional        geol_porosity     parFC           3          -0.019189               0.019189               0.000145          0.012061          0.000514          1.312650e-07         0.000362         0.000005         0.000819          1.000000           0.000814
distributional   soil_depth_statsgo     parFC           3          -0.007930               0.011023               0.000210          0.014502          0.000273          1.431319e-07         0.000378         0.000003         0.000808          0.666667           0.000805
distributional    max_water_content     parFC           3          -0.012999               0.012999               0.000136          0.011675          0.000305          1.593812e-07         0.000399         0.000017         0.000870          1.000000           0.000853
 deterministic       geol_1st_class     parFC           3           0.022522               0.022522               0.000147          0.012142          0.000655          1.962839e-07         0.000443         0.000029         0.000992          1.000000           0.000963
distributional               p_mean     parFC           3          -0.107319               0.107319               0.000005          0.002191          0.011522          2.186830e-07         0.000468         0.010880         0.011980          1.000000           0.001100
    mc_dropout  glim_2nd_class_frac     parFC           3           0.017992               0.018884               0.000223          0.014919          0.000546          2.576475e-07         0.000508         0.000002         0.001224          0.666667           0.001222
 deterministic  glim_1st_class_frac     parFC           3          -0.024049               0.024049               0.000191          0.013827          0.000770          2.905697e-07         0.000539         0.000021         0.001270          1.000000           0.001249
 deterministic        geol_porosity     parFC           3          -0.018275               0.018275               0.000180          0.013427          0.000514          3.805311e-07         0.000617         0.000062         0.001386          1.000000           0.001324

## Key-parameter correlation stability overview

parameter        variant  mean_spearman_r2  mean_variance_spearman_r2  mean_sign_consistency  topk_overlap_rate
  parBETA  deterministic          0.102327                   0.000551               0.971429           0.466667
  parBETA distributional          0.097121                   0.000044               1.000000           0.866667
  parBETA     mc_dropout          0.100319                   0.002134               0.933333           0.600000
 parCFMAX  deterministic          0.088385                   0.000809               0.971429           0.666667
 parCFMAX distributional          0.039502                   0.000292               0.980952           0.666667
 parCFMAX     mc_dropout          0.109854                   0.000416               1.000000           0.733333
    parFC  deterministic          0.056161                   0.000097               0.980952           1.000000
    parFC distributional          0.052056                   0.000029               0.980952           1.000000
    parFC     mc_dropout          0.054614                   0.000661               0.971429           0.866667
  parPERC  deterministic          0.099652                   0.000151               0.980952           1.000000
  parPERC distributional          0.094787                   0.000049               0.980952           1.000000
  parPERC     mc_dropout          0.109420                   0.000529               0.942857           0.866667
   parUZL  deterministic          0.084754                   0.000031               0.971429           1.000000
   parUZL distributional          0.085729                   0.000086               0.971429           1.000000
   parUZL     mc_dropout          0.090402                   0.000249               0.980952           0.866667
  route_b  deterministic          0.063996                   0.000125               0.971429           0.866667
  route_b distributional          0.056849                   0.000490               0.942857           0.600000
  route_b     mc_dropout          0.065243                   0.000161               0.980952           1.000000

## Cross-method consistency across key parameters

parameter      variant_a      variant_b  attribute_count  pearson_corr_of_mean_rho  spearman_corr_of_mean_rho  mean_abs_diff_rho  sign_agreement_rate  topk_overlap_rate
  parBETA  deterministic distributional               35                  0.995932                   0.993277           0.023235             0.971429                1.0
  parBETA  deterministic     mc_dropout               35                  0.976056                   0.965266           0.056606             0.914286                0.8
  parBETA distributional     mc_dropout               35                  0.989909                   0.982073           0.036696             0.942857                0.8
 parCFMAX  deterministic distributional               35                  0.643058                   0.617927           0.195804             0.600000                0.4
 parCFMAX  deterministic     mc_dropout               35                  0.980329                   0.966947           0.059510             0.914286                0.8
 parCFMAX distributional     mc_dropout               35                  0.694870                   0.714006           0.214032             0.628571                0.4
    parFC  deterministic distributional               35                  0.998564                   0.995798           0.012957             1.000000                1.0
    parFC  deterministic     mc_dropout               35                  0.991253                   0.976471           0.024543             0.971429                1.0
    parFC distributional     mc_dropout               35                  0.994484                   0.984314           0.017901             0.971429                1.0
  parPERC  deterministic distributional               35                  0.999192                   0.995798           0.012936             0.971429                1.0
  parPERC  deterministic     mc_dropout               35                  0.998549                   0.996078           0.017004             0.971429                1.0
  parPERC distributional     mc_dropout               35                  0.997449                   0.994398           0.024247             1.000000                1.0
   parUZL  deterministic distributional               35                  0.999019                   0.997479           0.010312             0.971429                1.0
   parUZL  deterministic     mc_dropout               35                  0.996772                   0.993838           0.020139             0.914286                0.8
   parUZL distributional     mc_dropout               35                  0.997810                   0.994678           0.018592             0.942857                0.8
  route_b  deterministic distributional               35                  0.993260                   0.987955           0.028744             0.914286                0.8
  route_b  deterministic     mc_dropout               35                  0.979559                   0.961905           0.039135             0.914286                0.6
  route_b distributional     mc_dropout               35                  0.983156                   0.972549           0.041676             0.885714                0.6

## Overall cross-method consistency summary

     variant_a      variant_b  parameter_count  mean_pearson_corr_of_mean_rho  mean_spearman_corr_of_mean_rho  mean_abs_diff_rho  mean_sign_agreement_rate  mean_topk_overlap_rate
 deterministic distributional                6                       0.938171                        0.931373           0.047331                  0.904762                0.866667
 deterministic     mc_dropout                6                       0.987086                        0.976751           0.036156                  0.933333                0.833333
distributional     mc_dropout                6                       0.942946                        0.940336           0.058857                  0.895238                0.766667
