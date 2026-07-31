# Research Limitations and Future Work

## Resolved in V5.0

### 1. Two-Regime Setting Replaced by Three Regimes
The original manuscript only estimated a two-state Markov-switching GARCH.
V5.0 implements a three-regime calm / turbulent / crisis specification with a
Hamilton filter and a softmax-parameterized transition matrix
(`model/ms_garch.py`).

### 2. Category-Specific Sentiment Heterogeneity
Monetary, industrial, employment, and geopolitical sentiment channels are now
estimated separately, and a likelihood-ratio test of equal category
coefficients is implemented (`model/category_heterogeneity.py`).

### 3. Unified Regime-Switching Jump-GARCH
V5.0 estimates regime switching and compound Poisson jumps jointly in one
likelihood (`model/msj_garch.py`), instead of comparing them as competing
specifications.

### 4. Real Industry Mortality Data
The synthetic Lee-Carter series was removed. Mortality validation now uses the
official China Life Insurance Mortality Tables CL(2000-2003) and CL(2010-2013),
disaggregated by gender and business line (`data/raw/cl_mortality_panel.csv`).

## Remaining Limitations

### 1. Limited News Sample Size
The sentiment corpus still contains only a small number of curated Chinese
financial headlines. Category-level daily scores for the full 2018-2026 sample
are therefore constructed through a reproducible market-implied proxy, which
shares a common market factor across categories. A topic-level LLM corpus
covering the full sample is required to sharpen identification.

### 2. Two Official Mortality Vintages
Only CL(2000-2003) and CL(2010-2013) are used in the current panel. The
mortality application therefore relies on a two-point improvement measurement;
adding CL(2025) and census-based life tables would allow a full Lee-Carter
time-series identification.

### 3. Jump Distribution Restrictions
The unified MSJ-GARCH assumes Gaussian jump sizes and a truncated jump-count
mixture. Richer distributions and non-Gaussian jump tails are natural
extensions.

### 4. Heuristic Source Weight Assignment
Source weights (regulatory 1.0, sector news 0.7, market flash 0.4) remain
domain-judgment based. Endogenous weight estimation is implemented for the
aggregate index, and a full topic-level extension is left for future work.

## Future Work

1. Expand the news corpus to topic-labelled daily headlines across 2018-2026
   and re-estimate category multipliers without the proxy.
2. Add CL(2025) and NBS census life tables to obtain a true mortality time
   series for Lee-Carter identification.
3. Extend MSJ-GARCH to three regimes and alternative jump distributions.
4. Extend the framework beyond CSI 300 to CSI 500/1000 and sector indices.
