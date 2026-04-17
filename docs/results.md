## 3. Results

---

## 3.1 Invariant regularization improves PUB under hydrologically distinct domain shifts

### 3.1.1 Leave-one-cluster-out experimental setup and evaluation targets

这一小节主要是把结果场景重新交代清楚，但不要写成方法复述。重点强调：

* 7 个有效水文行为簇
* 每次留出 1 簇作为 OOD 测试集
* 训练集其余 6 簇作为 VREx environments
* 比较对象：ERM-dPL-HBV vs Invariant-dPL-HBV
* 核心指标：测试流域 KGE

**建议图表：**

**Figure 1. Experimental design of hydrologically distinct PUB evaluation**
内容包括：

* 7 个有效簇 A–G 的示意图或流程图
* 每一折 train/test 的划分逻辑
* ERM 与 VREx 的 loss 形式简图
  这张图更像“结果导向的实验图”，不是纯方法图。

**Table 1. Summary of hydrological clusters used as environments**
列建议：

* Cluster ID
* Original cluster composition
* Number of basins
* Brief hydrological characterization（可选）

---

### 3.1.2 Overall OOD performance across held-out hydrological clusters

这是第一主结果，必须最硬。

要回答：

* VREx 是否整体提高 OOD KGE
* 是否减少跨簇性能波动
* 是否改善最难簇的表现

**建议图表：**

**Figure 2. OOD KGE across seven held-out clusters for ERM and VREx**
推荐做法：

* x 轴：留出簇 A–G
* y 轴：测试 basin KGE
* 每个簇画 ERM vs VREx 的箱线图/小提琴图/散点+中位数

核心要让人一眼看到：

* 哪些簇提升明显
* 哪些簇差异不大
* VREx 是否更稳

**Table 2. Summary statistics of OOD performance**
列建议：

* Model
* Mean KGE across folds
* Median KGE across folds
* Worst-fold median KGE
* Inter-fold std / IQR
* Mean across 5 repeats
  这张表很重要，因为它把“均值提升”和“鲁棒性提升”都放进来了。

---

### 3.1.3 Performance gains are concentrated in out-of-distribution rather than random PUB settings

这一节用来放“随机十折”的辅助结果，作用是封口，不是主菜。

要表达的是：

* VREx 的优势主要体现在 hydrologically distinct OOD setting
* 在随机 PUB 下，差异可能更小

**建议图表：**

**Figure 3. Comparison between random PUB and hydrologically distinct PUB settings**
推荐两个面板：

* Panel (a): random 10-fold 的 ERM vs VREx KGE 分布
* Panel (b): leave-one-cluster-out 的 ERM vs VREx KGE 分布

让读者看到：

* 随机划分下差异有限
* 真正困难的 domain shift 下 VREx 更有价值

**Table 3. Performance comparison under random and hydrologically distinct PUB settings**
列建议：

* Split type
* Model
* Mean KGE
* Median KGE
* Worst-group KGE
* Variability metric

---

## 3.2 Invariant regularization stabilizes learned attribute-to-parameter relationships

这是第二主结果，也是全文最关键的“解释链条”。

---

### 3.2.1 Global attribute-parameter associations learned by ERM and VREx

这一节先回答：
模型到底学到了什么关系？

这里放的是**参数均值**与流域属性的关系，不建议一开始就放参数方差。

**建议图表：**

**Figure 4. Global attribute-to-parameter association heatmaps for ERM and VREx**
内容：

* 行：流域属性
* 列：HBV 参数
* 热力值：Spearman rho（参数均值 vs 属性）
* 左右并排：ERM、VREx

这张图的作用：

* 让人看到 VREx 是否学出更清晰、更集中的关系结构
* 哪些参数最受属性控制
* 哪些属性反复出现

**Table 4. Top attribute-parameter correlations learned by ERM and VREx**
列建议：

* Rank
* Attribute
* Parameter
* rho (ERM)
* rho (VREx)
* Difference
* Sign consistency across repeats（如果已有）

这张表最好只列 top 10 或 top 15，主文不要过长。

---

### 3.2.2 Relationship stability across held-out hydrological environments

这是本节核心，不是“rho 更高”，而是“更稳定”。

建议你明确定义一个稳定性指标，比如：

* 跨 7 个 held-out experiments 的 rho 标准差
* 或符号一致率
* 或 relation reproducibility score

**建议图表：**

**Figure 5. Stability of attribute-parameter relationships across held-out experiments**
可选做法之一：

* Panel (a): ERM 稳定性热力图
* Panel (b): VREx 稳定性热力图
* 热力值可以是 `1 / std(rho)`、sign consistency、或稳定性评分

或者更直接一点：

**Figure 5. Cross-experiment variability of attribute-parameter correlations**

* x 轴：relation pairs
* y 轴：std of rho across held-out experiments
* ERM vs VREx 对比

如果只能选一个，我更建议**热力图**，更直观。

**Table 5. Stability metrics of key attribute-parameter relationships**
列建议：

* Attribute
* Parameter
* Mean rho (ERM)
* Std rho (ERM)
* Mean rho (VREx)
* Std rho (VREx)
* Sign consistency (ERM / VREx)

这张表是全文非常关键的一张表。

---

### 3.2.3 Comparison with hydrologically expected parameter controls

这一节把你的结果和已有机理认知、MARRMoT 或相关文献结果对接。

注意这里不是要“逐条证明完全一致”，而是要说：

* VREx 学到的关系是否更接近已有水文认知
* ERM 是否更容易出现不一致或翻转关系

**建议图表：**

**Figure 6. Comparison of selected learned relationships with hydrological expectations / MARRMoT-informed controls**
做法建议：

* 选 4–6 组最关键关系
* 每组画 ERM 和 VREx 的散点图/趋势图
* 图中标注 literature-expected direction

比如：

* `frac_snow → parPERC`
* `low_prec_dur → parFC`
* `gvf_diff → parLP`
* `gvf_max → parK0`

**Table 6. Hydrological interpretation of selected attribute-parameter relationships**
列建议：

* Attribute
* Parameter
* Expected direction from prior understanding
* ERM direction / strength
* VREx direction / strength
* Interpretation

这张表很适合 WRR/HESS 风格。

---

## 3.3 Case studies reveal how VREx reshapes key hydrological parameters

这一节是深挖，但要克制。
建议只挑 2–4 个参数，不要所有参数都讲。

---

### 3.3.1 Soil-water storage parameter responses to climatic and vegetation gradients

优先给 `parFC`，因为你现在它最突出。

这里讨论：

* 哪些属性共同作用于 FC
* ERM vs VREx 的关系是否更平滑、更一致
* 是否跨环境更稳定

**建议图表：**

**Figure 7. Case study of parFC: response to selected climatic and vegetation attributes**
可以用 2–3 个面板：

* 单变量散点图：`low_prec_dur vs parFC_mean`
* 单变量散点图：`gvf_diff vs parFC_mean`
* 二维联合作用图：`low_prec_dur × gvf_diff -> parFC_mean`

注意这类图主文里不要太多，1 个参数做 2–3 幅就够。

---

### 3.3.2 Snow-related controls on percolation and runoff generation parameters

如果 `frac_snow` 很强，这节很自然。

讲：

* `frac_snow` 对 `parPERC`, `parBETA`, `parFC` 的影响
* 在不同 held-out snow / non-snow environments 下是否稳定

**建议图表：**

**Figure 8. Case study of snow-related controls on key parameters**
面板建议：

* `frac_snow vs parPERC_mean`
* `frac_snow vs parBETA_mean`
* 分簇对比图：不同 cluster 内该关系的 slope / rho

---

### 3.3.3 Uncertainty in parameter mapping varies systematically with basin attributes

这一节才放**参数方差与属性的关系**。
注意它是“增强分析”，不是主文最核心。

这里要表述清楚：

* MC dropout variance 是 parameter predictive uncertainty proxy
* 它反映模型对哪些 basin 类型更不确定
* VREx 是否降低了这种不确定性或改变了其分布

**建议图表：**

**Figure 9. Attribute controls on predictive uncertainty of selected parameters**
可做：

* 方差热力图：属性 vs 参数方差的 Spearman rho
* 或挑 2–3 个关键参数做属性–方差关系散点图

如果空间不够，这张图也可以移到 Appendix。

**Table 7. Summary of attribute–uncertainty associations for selected parameters**
列建议：

* Parameter
* Attribute
* rho with predictive mean
* rho with predictive variance
* Interpretation of uncertainty pattern

---

## 3.4 Robustness of the findings to alternative environment definitions

这一节建议简短，但很重要。它的存在会让审稿人更安心。

---

### 3.4.1 Results under USGS-based hydrological regionalization

这里就是你说的 USGS 分区。

要表达：

* 主要结论不依赖于 Gnann 聚类这一种环境定义

**建议图表：**

**Figure 10. OOD performance under alternative environment definition (USGS regions)**
内容：

* ERM vs VREx 在 USGS 分区 leave-one-region-out 下的 KGE 对比

---

### 3.4.2 Consistency of relationship stabilization under alternative region definitions

这节对应结果2的稳健性版本。

**建议图表：**

**Figure 11. Stability of learned attribute-parameter relationships under alternative region definitions**
内容：

* USGS 环境定义下的 relation stability heatmap
* 或稳定性统计汇总图

---

# Appendix 建议补充内容

附录的原则是：
**主文讲主线，附录负责完整性、稳健性、复现实验细节。**

---

## Appendix A. Additional details of hydrological environment construction

### A.1 Original 10-cluster to 7-cluster merging scheme

**Table A1. Mapping from original Gnann clusters to effective clusters**

### A.2 Basin counts and attribute summaries by cluster

**Table A2. Descriptive statistics of basin attributes by cluster**

### A.3 Spatial distribution of clusters

**Figure A1. Map of basin locations colored by hydrological cluster**

---

## Appendix B. Additional performance diagnostics

### B.1 Fold-by-fold results for all repeats

**Table A3. OOD KGE for each fold and each seed**

### B.2 Additional metrics beyond KGE

如果你还有 NSE、logNSE、bias 或 FHV/FLV 等，可放这里。
**Table A4. Additional performance metrics for ERM and VREx**

### B.3 Training dynamics

这部分很重要，可以放你现在观察到的 ramp 行为。

**Figure A2. Training curves of ERM term, VREx penalty term, and total loss**
最好分面板展示：

* mean loss
* variance penalty
* lambda ramp
* per-environment loss（可选）

---

## Appendix C. Additional analyses of attribute-to-parameter relationships

### C.1 Full correlation tables

**Table A5. Full Spearman correlation matrix for ERM**
**Table A6. Full Spearman correlation matrix for VREx**

### C.2 Repeat-wise stability of top relationships

**Table A7. Mean, std, and sign consistency across seeds for top relations**

### C.3 Cluster-wise correlation results

**Figure A3. Cluster-wise Spearman correlations for selected key relations**
这张图很重要，因为它能看出全局高相关是不是被少数簇拉动。

### C.4 Sign consistency maps

**Figure A4. Sign consistency of attribute-parameter relationships across experiments**

---

## Appendix D. Additional case studies on selected parameters

### D.1 Additional single-attribute response plots

**Figure A5. Scatterplots of selected attributes vs predicted parameter means**

### D.2 Joint effects of selected attribute pairs

**Figure A6. Two-dimensional response surfaces for selected parameter predictions**

### D.3 ERM vs VREx comparison for parameter uncertainty

**Figure A7. Comparison of predictive variance distributions across models**

---

## Appendix E. MC dropout convergence and uncertainty analysis

这一部分我建议一定加，因为你前面已经意识到 MC 次数可能影响方差分析。

### E.1 Convergence of parameter mean estimates with increasing MC samples

**Figure A8. Convergence of parameter mean statistics under MC = 10, 20, 50, 100**

### E.2 Convergence of parameter variance estimates with increasing MC samples

**Figure A9. Convergence of parameter variance statistics under MC = 10, 20, 50, 100**

### E.3 Stability of mean- and variance-based Spearman correlations

**Table A8. Top attribute-parameter correlations under different MC sample sizes**
这张表可以非常有力地说明：

* 均值相关是否很早就稳定
* 方差相关是否需要更多前向次数

---

## Appendix F. Robustness checks

### F.1 Results under alternative environment definitions

**Table A9. Performance summary under Gnann vs USGS region definitions**

### F.2 Sensitivity to lambda and warmup settings

如果你最终做了超参数敏感性分析，很值得放。
**Figure A10. Sensitivity of OOD KGE and relationship stability to lambda and warmup**

### F.3 Sensitivity to training epochs

**Figure A11. Performance and stability under 70 / 85 / 100 epochs**

---

# 一个更精炼的主文图表清单

如果你想控制主文篇幅，我建议主文图表数量大致为：

## 主文 Figures

1. 实验设计图
2. 7-fold OOD KGE 对比
3. 随机十折 vs OOD 对比
4. ERM vs VREx 的全局属性–参数热力图
5. 关系稳定性热力图
6. 与已有机理/MARRMoT 对照的关键关系图
7. `parFC` 案例图
8. snow-related 参数案例图
9. 参数方差–属性关系图
10. USGS 分区稳健性性能图
11. USGS 分区稳健性关系稳定性图

## 主文 Tables

1. 簇概况表
2. OOD 性能汇总表
3. 随机十折 vs OOD 性能汇总表
4. top 关系表
5. 关键关系稳定性表
6. 与已有机理对照表
7. 参数不确定性关系总结表

如果觉得太多，可以把 Figure 9、Figure 10、Figure 11 中的一部分移到附录。

---

# 我对主文篇幅的压缩建议

如果你希望更像 WRR 的紧凑写法，我建议主文保留：

* Figure 1
* Figure 2
* Figure 3
* Figure 4
* Figure 5
* Figure 6
* Figure 7
* Figure 8

以及：

* Table 1
* Table 2
* Table 4
* Table 5
* Table 6

剩余内容放 Appendix。

---

# 最后给你一个最推荐的三级标题版本

这是我认为目前最平衡的一版：

## 3. Results

### 3.1 Invariant regularization improves PUB under hydrologically distinct domain shifts

#### 3.1.1 Leave-one-cluster-out evaluation across hydrological behavior groups

#### 3.1.2 Overall OOD performance across held-out clusters

#### 3.1.3 Comparison with random PUB settings

### 3.2 Invariant regularization stabilizes learned attribute-to-parameter relationships

#### 3.2.1 Global attribute-parameter associations under ERM and VREx

#### 3.2.2 Cross-experiment stability of learned relationships

#### 3.2.3 Comparison with hydrological expectations and prior parameter controls

### 3.3 Case studies reveal how VREx reshapes key hydrological parameters

#### 3.3.1 Climatic and vegetation controls on soil-water storage parameters

#### 3.3.2 Snow-related controls on percolation and runoff generation parameters

#### 3.3.3 Attribute dependence of predictive uncertainty in parameter mapping

### 3.4 Robustness to alternative environment definitions

#### 3.4.1 OOD performance under USGS-based regionalization

#### 3.4.2 Stability of learned relationships under alternative region definitions
