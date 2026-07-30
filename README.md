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

`ash
git clone https://github.com/ffgh-code/insuranceandquantification.git
cd insuranceandquantification

pip install -r requirements.txt
streamlit run app/app.py
`

启动后终端会显示本地访问地址（通常是 http://localhost:8501），在浏览器打开即可。

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


## Deploy (Free, No Download Required)

The app is designed for **Streamlit Community Cloud** — deploy in 3 clicks, then share the link with anyone.

1. Go to **https://share.streamlit.io** and sign in with GitHub
2. Click **"New app"** → select ffgh-code/insuranceandquantification
3. Set **Main file path** to app/app.py and click **"Deploy"**

2 minutes later you get a public URL like https://insuranceandquantification.streamlit.app.

Put that link in your resume, portfolio, or Medium article so anyone can open it instantly.

Or deploy on any platform that supports Docker:

docker build -t sentiment-vol-lab .
docker run -p 8501:8501 sentiment-vol-lab
