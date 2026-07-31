# Insurance and Quantification

[![Streamlit App](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://insuranceandquantification-fbmyvwetabtki2r5m8krac.streamlit.app/)
[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

一个把 **LLM 情绪分析** 与 **GARCH/LSTM 波动率模型** 结合起来的量化项目，跑在**沪深 300 真实行情**上。项目同时包含三个精算扩展模块：Solvency II / C-ROSS 偿付能力、动态损失准备金、GARCH 增强死亡率预测。

---

## 项目简介与架构

```
Financial News / 财经新闻
        |
        v
  +------------------+
  | Sentiment Module  |  <- LLM + 金融词典回退 + VADER 基线
  | 来源加权 / 正负拆分 |     Granger 因果检验 / 4类话题分类
  +------------------+
        |
        v
  +------------------+     +------------------------+
  | Market Data       | --> | Volatility Models       |
  | akshare 沪深300   |     | RV / GARCH / ARIMA /    |
  | 5分钟 + 日线       |     | LSTM / Transformer     |
  +------------------+     +------------------------+
        |                          |
        v                          v
  +------------------+     +------------------------+
  | Strategy Layer    |     | Actuarial Module        |
  | 滚动回测 240天月滚 |     | SCR 双框架 / 损失准备金 /|
  | 横截面多因子选股   |     | Lee-Carter+GARCH        |
  | 牛熊震荡分区       |     +------------------------+
  +------------------+
        |
        v
  +------------------+
  | Streamlit Dashboard |  <- 在线 Demo
  +------------------+
```

## 环境部署

```bash
# 1. 克隆仓库
git clone https://github.com/ffgh-code/insuranceandquantification.git
cd insuranceandquantification

# 2. 安装依赖（Python 3.13+）
pip install -r requirements.txt

# 3. 启动本地 Dashboard
streamlit run app/app.py
```

启动后终端会显示本地访问地址（通常是 http://localhost:8501），在浏览器打开即可。

## 各模块启动指令

| 模块 | 指令 | 说明 |
|------|------|------|
| Streamlit Dashboard | `streamlit run app/app.py` | 10 个页面 |
| Jupyter 分析 | `jupyter notebook notebooks/01_full_analysis.ipynb` | 完整分析报告 |
| 单元测试 | `pytest tests/ -v` | 核心 + 精算模块测试 |
| 研究报告 | `docs/research_report.md` | 9 章节研究文稿 |
| 大创申报 | `docs/proposal_300.md` | 300字课题申报简介 |

## 在线 Demo

**https://insuranceandquantification-fbmyvwetabtki2r5m8krac.streamlit.app/**

无需安装，浏览器直接打开。

## 模块说明

### data（数据源）
- `market_data.py`：akshare 沪深300 + yfinance 美股 + 合成数据兜底，双分支设计
- `news_scraper.py`：RSS 抓取 + 中文财经头条样本

### sentiment（情绪模块）
- `llm_sentiment.py`：OpenAI 兼容 API + 金融词库回退
- `traditional_sentiment.py`：VADER / TextBlob 基线
- `sentiment_agg.py`：来源加权聚合、正负情绪拆分、Granger 因果、4类话题 GARCH-X

### volatility（时序建模）
- `realized_vol.py`：Close-to-Close / Parkinson / Garman-Klass / Yang-Zhang
- `garch.py`：GARCH / EGARCH / GJR-GARCH / GARCH-X
- `arima_model.py`：ARIMA 基线 + Ljung-Box 残差检验
- `lstm_vol.py`：LSTM（PyTorch）
- `transformer_model.py`：单层注意力 Transformer
- `highfreq.py`：5分钟高频数据 + 日内已实现波动率（双分支）

### backtest（回测）
- `backtest.py`：基础回测引擎（交易成本 0.1%，滑点 0.05%）
- `rolling_backtest.py`：滚动窗口回测（240天月滚 7:3）、行情分区、月收益直方图
- `cross_sectional.py`：沪深300成分股多因子选股、IC/IR

### actuary（精算工具）
- `solvency.py`：Solvency II + C-ROSS 双框架 SCR 计算、压力测试
- `loss_modeling.py`：静态 vs 动态准备金
- `mortality.py`：朴素 / Lee-Carter / GARCH-LC 三模型对照

## 版本迭代记录

| 版本 | 阶段 | 内容 |
|------|------|------|
| V1.0 | 原型 | 基础 pipeline：沪深300日线 + LLM/VADER 情绪 + GARCH/LSTM + 4策略回测 |
| V2.0 | 短期优化 | 中文新闻加权、正负情绪拆分、滚动窗口回测、横截面多因子、行情分区 |
| V3.0 | 中期完整版 | 5分钟高频 RV、ARIMA/Transformer、精算三模块重构、config.yaml、研究报告 |

## 目录结构

```
insuranceandquantification/
|-- app/               # Streamlit Dashboard + Pipeline
|-- data/scrapers/     # 数据源（akshare/yfinance/RSS）
|-- sentiment/         # 情绪模块
|-- volatility/        # 时序建模模块
|-- strategy/          # 回测模块
|-- actuarial/         # 精算工具模块
|-- docs/              # 研究报告 / 大创申报 / PDF
|-- tests/             # 单元测试
|-- config.yaml        # 全局配置
|-- requirements.txt
|-- README.md
```

## License

MIT
