# Insurance and Quantification

A quantitative finance project combining **LLM-powered sentiment analysis** with **statistical and deep learning volatility models** (GARCH + LSTM), running on **real Chinese A-share market data (CSI 300 index)**.

The modular architecture makes it easy to extend with US market data, custom strategies, and actuarial applications.

---

## Key Features

- **Real Chinese Market Data** — 沪深300 (CSI 300) index data via akshare, with US yfinance and synthetic fallback
- **Chinese Financial News Sentiment** — LLM-powered + financial lexicon fallback for A-share market news
- **Multiple Volatility Estimators** — Close-to-Close, Parkinson, Garman-Klass, Yang-Zhang
- **GARCH Family Models** — GARCH, EGARCH, GJR-GARCH with GARCH-X (exogenous sentiment)
- **LSTM Volatility Prediction** — PyTorch-based with configurable architecture
- **Strategy Backtesting** — 4 strategies with full risk metrics
- **Actuarial Applications** — Solvency II / C-ROSS, loss reserving, mortality forecasting

---

## How It Works

```
Chinese Financial News -> LLM Sentiment Analysis
                            |
CSI 300 Index Data -> Realized Volatility -> GARCH + LSTM Models
                            |
Sentiment + Volatility -> Trading Strategy -> Backtest
```

---

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

Open http://localhost:8501 in your browser.

---

## Results (CSI 300)

| Component | Result |
|-----------|--------|
| Market Data | 727 trading days of real CSI 300 data |
| Sentiment | 20 Chinese financial headlines analyzed |
| GARCH(1,1) | AIC: 2,030 (valid fit) |
| Best Strategy | Combined (Sharpe: 0.43) |

---

## Project Structure

```
sentiment-vol-lab/
|-- app/                  # Streamlit dashboard + pipeline
|-- data/scrapers/        # Market data (akshare + yfinance) + news
|-- sentiment/            # LLM + VADER sentiment analysis
|-- volatility/           # Realized vol, GARCH, LSTM
|-- strategy/             # Backtesting engine + trading strategies
|-- actuarial/            # Solvency II, loss reserving, mortality
|-- notebooks/            # Jupyter analysis notebook
|-- tests/                # 36 unit tests
```

---

## Tech Stack

| Category | Libraries |
|----------|-----------|
| Data | akshare, yfinance, BeautifulSoup |
| ML/AI | PyTorch, scikit-learn |
| Statistics | arch (GARCH), statsmodels |
| Sentiment | OpenAI API, VADER, TextBlob |
| UI | Streamlit, Plotly |
