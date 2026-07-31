"""Streamlit dashboard.

Run with: streamlit run app/app.py
"""

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.pipeline import Pipeline

st.set_page_config(
    page_title="Sentiment-Enhanced Volatility Lab",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    "<style>#MainMenu{visibility:hidden;}footer{visibility:hidden;}.stDeployButton{display:none;}</style>",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_pipeline():
    return Pipeline(ticker="sh000300")


@st.cache_data(ttl=3600)
def run_pipeline():
    pipeline = get_pipeline()
    results = pipeline.run_all()
    return results, pipeline


def main():
    st.sidebar.image(
        "https://img.icons8.com/fluency/96/stock-exchange.png",
        width=60,
    )
    st.sidebar.title("Sentiment Vol Lab")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        [
            "Overview",
            "Market Data",
            "Sentiment Analysis",
            "Volatility Models",
           "Strategy Backtest",
            "Actuarial Applications",
           "Full Pipeline",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Ticker:** sh000300 (CSI 300)")
    st.sidebar.markdown("**Data:** A-share data (akshare) + Chinese financial news")
    st.sidebar.markdown("**Sentiment:** LLM + VADER (OpenAI optional)")
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Last update: {datetime.now():%Y-%m-%d %H:%M}")

    with st.spinner("Running full analysis pipeline..."):
        results, pipeline = run_pipeline()

    pages = {
        "Overview": lambda: show_overview(results, pipeline),
        "Market Data": lambda: show_market_data(pipeline),
        "Sentiment Analysis": lambda: show_sentiment(pipeline),
        "Volatility Models": lambda: show_volatility(pipeline),
       "Strategy Backtest": lambda: show_strategy(pipeline),
        "Actuarial Applications": lambda: show_actuarial(pipeline),
       "Full Pipeline": lambda: show_full_pipeline(results),
    }
    pages[page]()


def show_overview(results, pipeline):
    st.title("Sentiment-Enhanced Volatility Prediction Platform")
    st.markdown(
        "A quant finance project combining **LLM sentiment analysis** "
        "with **volatility models** (GARCH + LSTM) for trading signals."
    )

    d = results["data"]
    s = results["sentiment"]
    v = results["volatility"]
    st2 = results["strategy"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Price (SPY)", f"${d['current_price']:.2f}")
    c2.metric("Current Vol (Ann.)", f"{d['current_vol']:.1%}")
    c3.metric("Sentiment Score", f"{s['llm_avg_sentiment']:.3f}")
    c4.metric("Best Sharpe", f"{st2['best_sharpe']:.2f}")

    st.subheader("Architecture")
    col1, col2 = st.columns(2)
    col1.code(
        "Financial News -> LLM Sentiment\n"
        "                   |\n"
        "Market Data -> GARCH + LSTM\n"
        "                   |\n"
        "Sentiment + Vol -> Strategy -> Backtest"
    )
    col2.markdown(
        "- **Sentiment:** LLM + rule-based + VADER\n"
        "- **Volatility:** Parkinson, GK, YZ estimators\n"
        "- **Models:** GARCH(1,1), EGARCH, GJR-GARCH, LSTM\n"
        "- **Strategies:** Vol mean reversion, sentiment, combined"
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Market Data", f"{d['n_observations']:,} days")
    c2.metric("Headlines", str(s["n_headlines"]))
    c3.metric("Strategies", str(st2["n_strategies"]))


def show_market_data(pipeline):
    st.title("Market Data & Realized Volatility")
    prices = pipeline._prices
    rv = pipeline._realized_vol
    if prices is None or prices.empty:
        st.warning("No data.")
        return

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=prices.index, open=prices["open"], high=prices["high"],
        low=prices["low"], close=prices["close"], name="SPY"
    ))
    fig.update_layout(height=450, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Realized Volatility Estimators")
    fig2 = go.Figure()
    for c in rv.columns:
        fig2.add_trace(go.Scatter(
            x=rv.index, y=rv[c], mode="lines",
            name=c.replace("_", " ").title()
        ))
    fig2.update_layout(height=350, yaxis_title="Ann. Volatility")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Latest Estimates")
    latest = rv.tail(5).round(4)
    latest.index = latest.index.strftime("%Y-%m-%d")
    st.dataframe(latest, use_container_width=True)


def show_sentiment(pipeline):
    st.title("Financial Sentiment Analysis")
    st.markdown(
        "Comparing **LLM-powered** sentiment against **VADER** baseline."
    )
    llm = pipeline._llm_sentiment_df
    trad = pipeline._trad_sentiment_df
    if llm is None:
        st.warning("No data.")
        return

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=llm["llm_score"], nbinsx=20, name="LLM Score",
            marker_color="royalblue", opacity=0.7
        ))
        fig.add_trace(go.Histogram(
            x=trad["vader_compound"], nbinsx=20, name="VADER Score",
            marker_color="lightcoral", opacity=0.7
        ))
        fig.update_layout(title="Score Distribution", barmode="overlay", height=350)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        dc = llm["llm_direction"].value_counts()
        fig2 = go.Figure(data=[go.Pie(
            labels=dc.index, values=dc.values, hole=0.4,
            marker_colors=["#2ecc71", "#e74c3c", "#95a5a6"]
        )])
        fig2.update_layout(title="Direction Breakdown", height=350)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Topic Distribution")
    tc = llm["llm_topic"].value_counts()
    fig3 = px.bar(x=tc.index, y=tc.values, color=tc.values, color_continuous_scale="Viridis")
    fig3.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Sample Headlines")
    cols = {"headline": "Headline", "llm_score": "Score",
            "llm_direction": "Direction", "llm_topic": "Topic"}
    st.dataframe(llm[list(cols.keys())].head(10).rename(columns=cols),
                 use_container_width=True, hide_index=True)


def show_volatility(pipeline):
    st.title("Volatility Models: GARCH & LSTM")
    gr = pipeline._garch_result
    rv = pipeline._realized_vol
    if gr is None:
        st.warning("Not fitted yet.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("GARCH AIC", f"{gr['aic']:.1f}")
    c2.metric("GARCH BIC", f"{gr['bic']:.1f}")
    c3.metric("Log-Likelihood", f"{gr['loglikelihood']:.1f}")

    cv = gr["conditional_volatility"]
    if not cv.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cv.index, y=cv, mode="lines",
                                 name="GARCH Cond Vol", line=dict(color="royalblue")))
        fig.add_trace(go.Scatter(x=rv.index, y=rv["yang_zhang"], mode="lines",
                                 name="Realized Vol (YZ)", line=dict(color="orange", dash="dot")))
        fig.update_layout(height=350, yaxis_title="Ann. Volatility")
        st.plotly_chart(fig, use_container_width=True)

    lr = pipeline._lstm_result
    if lr and lr.get("final_val_loss"):
        st.subheader("LSTM Training")
        c1, c2 = st.columns(2)
        c1.metric("Train Loss", f"{lr.get('final_train_loss', 0):.6f}")
        c2.metric("Val Loss", f"{lr.get('final_val_loss', 0):.6f}")
        if lr.get("train_losses"):
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(y=lr["train_losses"], mode="lines", name="Train Loss"))
            fig2.add_trace(go.Scatter(y=lr["val_losses"], mode="lines", name="Val Loss"))
            fig2.update_layout(height=300, xaxis_title="Epoch", yaxis_title="Loss")
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("LSTM not trained in this quick run.")


def show_strategy(pipeline):
    st.title("Strategy Backtest Results")
    sr = pipeline._strategy_results
    if sr is None:
        st.warning("No backtest results.")
        return

    comp = sr["comparison"]
    combined = sr["combined_result"]

    st.subheader("Strategy Comparison")
    st.dataframe(comp, use_container_width=True, hide_index=True)

    if combined and combined["metrics"]:
        m = combined["metrics"]
        st.subheader("Combined Strategy Detail")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Return", f"{m.total_return_pct:.2f}%")
        c2.metric("Sharpe", f"{m.sharpe_ratio:.2f}")
        c3.metric("Max DD", f"{m.max_drawdown:.2%}")
        c4.metric("Win Rate", f"{m.win_rate:.1%}")

        if not m.equity_curve.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=m.equity_curve.index, y=m.equity_curve, mode="lines",
                name="Portfolio", line=dict(color="royalblue", width=2),
                fill="tozeroy", fillcolor="rgba(65,105,225,0.1)"
            ))
            fig.add_hline(y=1_000_000, line_dash="dash", line_color="gray",
                          annotation_text="Initial")
            fig.update_layout(height=350, yaxis_title="Portfolio Value ($)")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Detailed Metrics")
        st.dataframe(m.to_dataframe(), use_container_width=True, hide_index=True)


def show_full_pipeline(results):
    st.title("End-to-End Pipeline Results")

    d = results["data"]
    s = results["sentiment"]
    v = results["volatility"]
    st2 = results["strategy"]

    st.subheader("Summary")
    df = pd.DataFrame({
        "Component": ["Market Data", "Sentiment", "Volatility Models", "Strategy"],
        "Status": [
            f"{d['n_observations']} days",
            f"{s['n_headlines']} headlines",
            f"GARCH AIC={v['garch_aic']:.1f}",
            f"Best: {st2['best_strategy']} (S={st2['best_sharpe']:.2f})",
        ]
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("Technical Details")
    st.json({
        "data": {"ticker": d["ticker"], "obs": d["n_observations"],
                 "price": round(d["current_price"], 2),
                 "avg_vol": round(d["avg_vol"], 4)},
        "sentiment": {"llm_mean": round(s["llm_avg_sentiment"], 4),
                      "vader_mean": round(s["vader_avg_sentiment"], 4),
                      "bullish": round(s["llm_bullish_ratio"], 4)},
        "garch": {"aic": round(v["garch_aic"], 2), "bic": round(v["garch_bic"], 2)},
        "strategy": {"best": st2["best_strategy"],
                     "sharpe": round(st2["best_sharpe"], 4)},
    })

    if v.get("garch_comparison") is not None:
        st.subheader("GARCH Model Selection")
        st.dataframe(v["garch_comparison"], use_container_width=True, hide_index=True)

    st.success("Pipeline completed successfully!")




def show_actuarial(pipeline):
    st.title("Actuarial Science Applications")
    st.markdown(
        "Bridging quant finance volatility modeling with actuarial "
        "solvency capital, loss reserving, and mortality forecasting."
    )

    tab1, tab2, tab3 = st.tabs([
        "Solvency II / C-ROSS",
        "Loss Reserving",
        "Mortality Forecasting",
    ])

    with tab1:
        st.subheader("Solvency Capital Requirement (SCR)")
        st.markdown(
            "Using GARCH volatility forecasts to estimate market risk "
            "solvency capital under Solvency II (EU) and C-ROSS (China) frameworks."
        )

        from actuarial.solvency import SolvencyCalculator
        import numpy as np

        sc = SolvencyCalculator(capital_base=100_000_000)

        if pipeline._garch_result is not None:
            cond_vol = pipeline._garch_result.get("conditional_volatility")
            if cond_vol is not None and not cond_vol.empty:
                report = sc.generate_report(cond_vol, available_capital=120_000_000)

                c1, c2, c3 = st.columns(3)
                c1.metric("Current Vol", f"{report['current_vol']:.2%}")
                c2.metric("Solvency II SCR", f"")
                c3.metric("C-ROSS SCR", f"")

                if "capital_adequacy" in report:
                    c4, c5 = st.columns(2)
                    c4.metric("Capital Adequacy", f"{report['capital_adequacy']:.1f}%")
                    c5.metric("Peak SCR", f"")

                market_risk = sc.scratch_market_risk(cond_vol)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=market_risk.index, y=market_risk["scr_market"],
                    mode="lines", name="SCR",
                    line=dict(color="royalblue", width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=market_risk.index, y=market_risk["mcr_market"],
                    mode="lines", name="MCR",
                    line=dict(color="orange", width=1.5, dash="dot")
                ))
                fig.update_layout(height=300, yaxis_title="Capital ($)")
                st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            vol_input = st.slider("Annualized Vol", 0.05, 0.60, 0.20, 0.01, key="scr_vol")
        with c2:
            cap_input = st.number_input("Exposure ($)", 1_000_000, 1_000_000_000, 100_000_000, step=1_000_000, key="scr_cap")

        var_995 = SolvencyCalculator.calculate_var(vol_input, 0.995)
        var_99 = SolvencyCalculator.calculate_var(vol_input, 0.99)
        cvar_995 = SolvencyCalculator.calculate_cvar(vol_input, 0.995)

        c1, c2, c3 = st.columns(3)
        c1.metric("VaR (99.5%)", f"")
        c2.metric("VaR (99%)", f"")
        c3.metric("CVaR (99.5%)", f"")

    with tab2:
        st.subheader("Insurance Loss Reserving with GARCH")
        st.markdown(
            "Applying volatility models to insurance claim data for loss reserving."
        )

        from actuarial.loss_modeling import LossModeler
        lm = LossModeler()
        claim_data = lm.generate_claim_data()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=claim_data["date"], y=claim_data["total_loss"],
            name="Total Loss", marker_color="royalblue", opacity=0.7
        ))
        fig.add_trace(go.Scatter(
            x=claim_data["date"], y=claim_data["cumulative_paid"],
            mode="lines+markers", name="Cumulative Paid",
            line=dict(color="orange", width=2)
        ))
        fig.update_layout(height=300, yaxis_title="Loss ($)")
        st.plotly_chart(fig, use_container_width=True)

        comparison = lm.compare_reserving_methods(claim_data)
        st.subheader("Reserving Method Comparison")
        st.dataframe(comparison, use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("Mortality Improvement Rate Forecasting")
        st.markdown(
            "Comparing Lee-Carter, GARCH, and naive mortality forecasting methods."
        )

        from actuarial.mortality import MortalityForecaster
        mf = MortalityForecaster()
        mort_data = mf.generate_mortality_data(n_years=50)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=mort_data["date"], y=mort_data["mortality_rate"],
            mode="lines", name="Mortality Rate",
            line=dict(color="royalblue", width=2)
        ))
        fig.add_trace(go.Scatter(
            x=mort_data["date"], y=mort_data["volatility"],
            mode="lines", name="Volatility",
            line=dict(color="orange", width=1.5, dash="dot"),
            yaxis="y2"
        ))
        fig.update_layout(
            height=300,
            yaxis_title="Mortality Rate",
            yaxis2=dict(title="Volatility", overlaying="y", side="right"),
        )
        st.plotly_chart(fig, use_container_width=True)

        lc_forecast = mf.lee_carter_forecast(mort_data["log_mortality"], 10)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=mort_data["date"], y=mort_data["mortality_rate"],
            mode="lines", name="Historical",
            line=dict(color="royalblue")
        ))
        forecast_dates = pd.date_range(
            start=mort_data["date"].iloc[-1] + pd.Timedelta(days=365),
            periods=10, freq="YE"
        )
        fig2.add_trace(go.Scatter(
            x=forecast_dates, y=lc_forecast["forecast_mortality"],
            mode="lines+markers", name="Forecast",
            line=dict(color="red", dash="dash")
        ))
        fig2.add_trace(go.Scatter(
            x=forecast_dates, y=lc_forecast["upper_95"],
            mode="lines", name="Upper 95%",
            line=dict(color="red", width=0), showlegend=False
        ))
        fig2.add_trace(go.Scatter(
            x=forecast_dates, y=lc_forecast["lower_95"],
            mode="lines", name="Lower 95%",
            line=dict(color="red", width=0), fillcolor="rgba(255,0,0,0.1)",
            fill="tonexty", showlegend=False
        ))
        fig2.update_layout(height=300, yaxis_title="Mortality Rate")
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Method Comparison")
        comp = mf.compare_forecast_methods(mort_data, 10)
        st.dataframe(comp, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
