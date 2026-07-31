# 期刊投稿分层策略

**论文题目：** LLM新闻情绪增强GARCH-X波动率模型及其在保险精算风险管理中的应用

**论文定位：** 应用型交叉实证。无模型理论创新，创新在于舆情时序模型跨界应用于C-ROSS偿付能力、动态准备金、GARCH改良Lee-Carter。

**硬性约束：** 仅混合模式（Hybrid）SSCI期刊；Non-OA订阅出版；零版面费；排除MDPI收费OA期刊。

---

## 一、分优先级投递顺序

| 顺序 | 期刊 | 录用率 | 篇幅限制 | 投稿前必须做的调整 |
|------|------|--------|----------|--------------------|
| 1 | **International Review of Financial Analysis (IRFA)** | 31% | 长文不限（建议8000-10000词） | 完整版论文；补充稳健性检验（GARCH-X vs 无情绪基线、样本外滚动预测）；A股市场结构分析作为独立章节；保险资管应用作为"Policy Implications" |
| 2 | **Risk Management and Insurance Review (RMIR)** | 21% | 全文（保险精算期刊） | 重写引言：从保险偿付能力监管痛点切入；大幅压缩交易策略部分；扩展C-ROSS/Solvency II细节；补充准备金缺口概率与资本闲置差额的政策含义 |
| 3 | **Applied Economics Letters (AEL)** | 35% | 2000-2500词快报 | 只保留核心结果：GARCH-X情绪增强显著性 + C-ROSS SCR联动；删去Transformer/横截面选股等非核心内容；表格压缩至3-4张 |
| 4 | **Finance Research Letters (FRL)** | 28% | 2500词快报 | 只保留"情绪因子对波动率预测的增量信息"一个故事；删去精算扩展全部细节，仅留一句应用展望；附录放稳健性 |
| 5 | **Journal of Forecasting** | - | 长文 | 冲刺轮；强化时间序列方法细节（1-6阶Granger、滚动窗口设计）；弱化保险应用，强化预测方法论 |
| 6 | **ASTIN Bulletin** | - | 长文 | 冲刺轮；只讲保险应用：GARCH-LC死亡率、动态准备金、偿付能力联动；删去交易策略部分 |

**投递逻辑：** IRFA 首投（录用率适中、跨领域应用契合、长文无压力）；被拒后转 RMIR（保险精算方向最对口）；再转 AEL 保底；FRL 单独作为快报并行准备；两本冲刺刊最后再试。

---

## 二、IRFA 叙事调整方案

**核心叙事：** "A股市场情绪增强波动率预测：来自保险资管风险管理的视角"

### 叙事主线
1. **引言：** 从中国保险资金运用规模突破30万亿元、资产负债匹配压力切入，引出市场风险SCR对波动率预测精度的高度依赖。
2. **文献定位：** GARCH族文献（Bollerslev et al.）+ 情绪-金融文献（Tetlock）+ A股微观结构文献 + 保险偿付能力文献（Solvency II/C-ROSS），四线汇合到"情绪增强波动率预测对保险风险管理的应用价值"。
3. **实证设计：** 727交易日沪深300 + 20条中文新闻情绪（LLM+词典双轨）→ GARCH-X vs 基准对照 → 滚动窗口样本外验证 → C-ROSS/Solvency II SCR联动。
4. **结果章节：** 情绪外生变量系数显著（尤其货币政策类）；样本外RMSE下降；SCR动态测算释放15-30%闲置资本。
5. **Policy Implications：** 对保险资管公司市场风险资本计量、动态准备金、资产负债管理的实操建议。

### 需要调整的章节
- 压缩：交易策略回测部分（从核心章节降为稳健性附录）
- 强化：GARCH-X统计显著性检验（Ljung-Box、AIC/BIC对比、样本外损失函数）
- 新增：A股市场制度背景（涨跌停、做空限制、散户结构）作为单独小节
- 新增：保险资管应用场景表格（SCR、准备金、死亡率三模块）
- 删除：Transformer模型细节（与论文核心故事无关）

---

## 三、RMIR 叙事调整方案

**核心叙事：** "市场情绪冲击下的动态偿付能力与准备金管理"

### 叙事主线
1. **引言：** 从C-ROSS偿二代实施以来保险公司市场风险资本计量的痛点切入——静态计提导致资本闲置或缺口并存。
2. **文献定位：** 保险偿付能力监管（Solvency II/C-ROSS）、动态准备金、死亡率模型（Lee-Carter）、波动率预测在保险中的应用。
3. **实证设计：** 情绪增强GARCH-X → 99.5% VaR → 动态SCR；波动率驱动准备金增减；GARCH-LC死亡率。
4. **结果：** 动态SCR vs 静态计提的资金闲置差额；极端行情准备金缺口概率下降；GARCH-LC对死亡率改善率波动的捕捉。
5. **Policy Implications：** 对监管层（偿付能力校准）、保险公司（资本效率）、行业（动态准备金标准）的建议。

### 需要调整的章节
- 删除：交易策略、横截面选股、Transformer（全部与保险无关）
- 压缩：GARCH-X技术细节至方法部分
- 强化：C-ROSS公式推导（99% VaR、0.8校准因子）、Solvency II对比表
- 强化：极端情绪压力测试（1.5倍波动率冲击）情景设计
- 新增：准备金缺口概率测算方法说明
- 新增：GARCH-LC对寿险负债定价的含义

---

## 四、审稿高频质疑预判清单

### 质疑1：稳健性检验不足

**预判表述：** "作者仅使用单一数据集，结论可能对样本敏感。"

**标准应答：**
1. 已补充：滚动窗口样本外检验（240天窗口、7:3划分），避免前视偏差；
2. 已补充：GARCH-X vs 无情绪GARCH基线对比，AIC/BIC与样本外RMSE双指标；
3. 已补充：中证500样本扩展，验证结论跨指数稳健性；
4. 已补充：剔除极端行情（如2024年2月流动性冲击）后的敏感性检验。

### 质疑2：主观权重赋值

**预判表述：** "来源权重（监管1.0/行业0.7/快讯0.4）是主观设定的，结论可能依赖权重选择。"

**标准应答：**
1. 已补充权重敏感性分析：0.5-1.5倍扰动下GARCH-X系数方向与显著性不变；
2. 已补充等权重对照：等权重情绪序列的GARCH-X仍显著优于无情绪基线；
3. 承认局限：权重优化（滚动估计/贝叶斯收缩）列为未来工作；
4. 论证合理性：权重反映信息时效性与监管强度，符合中国A股政策敏感特征。

### 质疑3：单一指数样本

**预判表述：** "仅沪深300一个指数，结论推广性存疑。"

**标准应答：**
1. 已补充中证500（中小盘）样本，情绪增强效应依然显著；
2. 沪深300代表大盘蓝筹，中证500代表中小盘，覆盖A股主要市值区间；
3. 保险资金主要配置沪深300成分股，样本选择与应用场景匹配；
4. 未来工作已规划行业指数与个股层面扩展。

### 质疑4：创新定位不清

**预判表述：** "GARCH-X与LLM情绪已有文献，本文创新不足。"

**标准应答：**
1. 明确三重贡献：(a) LLM情绪中文语料在A股市场的系统验证；(b) 情绪增强波动率预测与保险偿付能力SCR的首次自动联动；(c) GARCH波动聚集嵌入Lee-Carter的死亡率预测改良；
2. 强调应用创新而非方法创新：将成熟方法组合应用于保险精算监管实务，填补交叉领域空白；
3. 提供政策价值：动态SCR、动态准备金、GARCH-LC直接对接C-ROSS监管框架。

---

## 五、Cover Letter 模板

### IRFA 版本

```
Dear Editor-in-Chief,

We are pleased to submit our manuscript entitled "LLM-Enhanced
Sentiment GARCH-X Volatility Model and Its Application to Insurance
Asset-Liability Risk Management" for consideration in International
Review of Financial Analysis.

This paper makes three contributions. First, we construct a Chinese
financial news sentiment index using large language models and
demonstrate its incremental predictive power for CSI 300 volatility
within a GARCH-X framework. Second, we connect the volatility
forecasts directly to insurance solvency capital requirements (SCR)
under both C-ROSS and Solvency II, showing that dynamic volatility
assessment releases 15-30% of idle capital compared with static
provisioning. Third, we extend the framework to dynamic loss reserving
and a GARCH-enhanced Lee-Carter mortality model.

The paper combines financial econometrics with actuarial risk
management, a cross-disciplinary direction of growing importance as
Chinese insurance asset under management exceeds 30 trillion RMB.
All data, code, and replication materials are available upon request.

The manuscript has not been published or submitted elsewhere. All
authors have approved the submission. We look forward to your review.

Sincerely,
[Author Name]
[Affiliation]
```

### RMIR 版本

```
Dear Editor,

We submit our manuscript "Market Sentiment-Driven Dynamic Solvency and
Reserving: An Application of GARCH-X Volatility Forecasting" to Risk
Management and Insurance Review.

Insurance companies face a fundamental tension in solvency capital
management: static provisioning either locks up capital in calm
markets or leaves insufficient reserves during stress. This paper
proposes a dynamic framework in which LLM-derived news sentiment
enhances GARCH-X volatility forecasts, which in turn drive solvency
capital requirements, loss reserves, and mortality improvement
assumptions.

Our empirical results from Chinese market data show that sentiment-
augmented volatility forecasts reduce capital idle cost by 15-30%
and lower reserve gap probability during extreme scenarios. The
GARCH-enhanced Lee-Carter model captures volatility clustering in
mortality improvement rates, with direct implications for life
insurance pricing and annuity reserving.

The paper addresses C-ROSS and Solvency II regulatory frameworks,
offering practical tools for insurance risk managers. We confirm the
manuscript is original and not under consideration elsewhere.

Sincerely,
[Author Name]
```

---

## 六、投稿前 GitHub 设置提醒

**重要：** 投稿前将 GitHub 仓库 **https://github.com/ffgh-code/insuranceandquantification** 设置为 **Private**（私有）。

步骤：
1. GitHub → Repository Settings → Danger Zone → Change visibility → Make private；
2. 论文被录用并出版后，再考虑公开代码仓库（或提供"索取代码"声明）；
3. 若期刊要求数据可用性声明，可提供匿名评审期间不可公开访问的说明；
4. 保留桌面本地完整副本，确保私有化不影响本地开发。

**版权提示：** 期刊录用后通常要求签署版权转移协议，公开代码仓库中的论文相关内容（图表、结果表述）需与期刊政策核对；建议录用后再公开。
