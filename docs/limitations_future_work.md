# Research Limitations and Future Work

## Limitations

### 1. Limited News Sample Size
The sentiment corpus comprises only 20 curated Chinese financial headlines per evaluation cycle. This constrained sample size limits the statistical power of Granger causality tests and may understate the true predictive relationship between news sentiment and volatility dynamics. A larger corpus spanning multiple years would provide more robust parameter estimates and allow sub-period stability checks.

### 2. Heuristic Source Weight Assignment
The source weights (regulatory 1.0, sector news 0.7, market flash 0.4) are assigned based on domain judgment rather than empirically estimated. This introduces subjective calibration into the sentiment aggregation pipeline. The optimal weights may vary across market regimes and policy cycles, and the current fixed-weight design cannot adapt to these shifts.

### 3. Single-Index Sample (CSI 300 Only)
All empirical results are based on the CSI 300 index alone. While the CSI 300 represents the broad large-cap A-share market, its volatility dynamics may not generalize to small-cap indices (CSI 500, CSI 1000), sector indices, or individual stocks. Cross-sectional factor analysis uses constituent stocks, but the volatility-strategy backtest remains index-level.

## Future Work

### 1. Alternative Data Expansion
Beyond structured news headlines, future iterations will incorporate alternative data sources including: earnings call transcripts for listed companies, patent filings for industrial policy tracking, satellite and traffic data for macro activity proxies, and social media sentiment (Weibo, Xueqiu) for retail investor behavior. These sources offer orthogonal information to traditional news and may improve volatility forecast accuracy during information-poor periods.

### 2. Dynamic Weight Optimization
Replace heuristic source weights with data-driven optimization. Candidate methods include: rolling-window maximum likelihood estimation of topic-specific sentiment-return sensitivities, Bayesian shrinkage toward empirical priors, and online learning frameworks that adapt weights to prevailing market regimes. The objective is a sentiment aggregation scheme that automatically up-weights regulatory signals during policy-sensitive periods and down-weights noisy retail chatter during calm markets.

### 3. Multi-Asset and Multi-Regime Extensions
Extend the framework beyond CSI 300 to include CSI 500/1000, sector indices, and cross-market comparisons (e.g., Hang Seng Tech, Nikkei). Multi-regime volatility models (Markov-switching GARCH) and regime-dependent sentiment weights will be explored to capture structural breaks in Chinese market volatility dynamics.
