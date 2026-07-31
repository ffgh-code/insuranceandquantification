# Repository Folder Structure

```
insuranceandquantification/
|
|-- app/                    # Streamlit Dashboard（10页面，五大板块）
|-- data/                   # 数据源：akshare/yfinance/RSS + 本地情绪CSV + 官方生命表
|-- model/                  # 时序建模：GARCH/ARIMA/LSTM/Transformer/MS-GARCH/MSJ-GARCH
|-- backtest/               # 回测引擎：滚动窗口/横截面/行情分区/对照组指标
|-- actuarial_calc/         # 精算工具：SCR/准备金/死亡率 + 误差指标
|-- sentiment_extract/      # 情绪模块：LLM/加权聚合/Granger/Qwen本地
|-- notebook/               # Jupyter分析Notebook
|-- docs/                   # 研究报告/局限性/未来展望/演示脚本
|-- tests/                  # 单元测试
|-- config.yaml             # 全局配置
|-- CHANGELOG.md            # 五次迭代日志
|-- Dockerfile              # 一键部署
|-- README.md
```

## 各文件夹说明

| 文件夹 | 用途 | 主要文件 |
|--------|------|----------|
| `app/` | Streamlit 交互面板，五大板块布局 | `app.py`, `pipeline.py` |
| `data/` | 数据获取与缓存（双分支设计） | `scrapers/market_data.py`, `sentiment_cache/local_sentiment.csv`, `raw/csi300_daily.csv`, `raw/cl_mortality_panel.csv` |
| `model/` | 波动率/收益率时序模型 | `ms_garch.py`, `msj_garch.py`, `category_heterogeneity.py`, `sentiment_proxy.py`, `garch.py`, `arima_model.py`, `lstm_vol.py`, `transformer_model.py`, `highfreq.py`, `realized_vol.py` |
| `backtest/` | 回测引擎与指标表 | `rolling_backtest.py`, `cross_sectional.py`, `control_group_metrics.py` |
| `actuarial_calc/` | 精算测算与误差指标 | `solvency.py`, `loss_modeling.py`, `mortality.py`, `error_metrics.py` |
| `sentiment_extract/` | 情绪提取与分析 | `llm_sentiment.py`, `sentiment_agg.py`, `local_sentiment.py`, `granger_full.py` |
| `notebook/` | 探索性分析与研究复现 | `01_full_analysis.ipynb` |
| `docs/` | 研究文档与全量估计结果 | `research_report.md`, `limitations_future_work.md`, `ime_extension_results.json` |

> 注意：本仓库仅存放量化代码、实证数据表、研究摘要、迭代日志与演示说明，不包含竞赛/大创申报类材料。
