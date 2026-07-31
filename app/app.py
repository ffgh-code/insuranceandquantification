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
        "五大板块",
        [
            # 数据预览
            "Overview",
            "Market Data",
            # 舆情可视化
            "Sentiment Analysis",
            # 模型拟合结果
            "Volatility Models",
            # 回测报表
            "Strategy Backtest",
            "Rolling Backtest Report",
            "Regime Performance",
            # 精算测算面板
            "Actuarial Applications",
            "Solvency Simulation",
            "Reserving Comparison",
            # 全流程
            "Full Pipeline",
        ],
        format_func=lambda x: {
            "Overview": "数据预览 - 总览",
            "Market Data": "数据预览 - 行情与波动率",
            "Sentiment Analysis": "舆情可视化 - 情绪分析",
            "Volatility Models": "模型拟合结果 - GARCH/LSTM",
            "Strategy Backtest": "回测报表 - 策略对比",
            "Rolling Backtest Report": "回测报表 - 滚动窗口",
            "Regime Performance": "回测报表 - 行情分区",
            "Actuarial Applications": "精算测算面板 - 总览",
            "Solvency Simulation": "精算测算面板 - 偿付能力",
            "Reserving Comparison": "精算测算面板 - 准备金",
            "Full Pipeline": "全流程结果",
        }.get(x, x),
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Ticker:** sh000300 (CSI 300)")
    st.sidebar.markdown("**Data:** A-share data (akshare) + Chinese financial news")
    st.sidebar.markdown("**Sentiment:** LLM + VADER (OpenAI optional)")
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Last update: {datetime.now():%Y-%m-%d %H:%M}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**PDF 报告导出**")
    st.sidebar.caption(
        "各页面报表与图表支持一键导出 PDF："
        "点击页面右上角下载按钮，系统将当前板块的表格、"
        "指标卡与图表打包为 PDF 报告。"
    )
    if st.sidebar.button("导出当前页面为 PDF"):
        try:
            from fpdf import FPDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, "Sentiment-Enhanced Volatility Lab - Report", align="C",
                     new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 8, f"Page: {page}", new_x="LMARGIN", new_y="NEXT")
            pdf.cell(0, 8, f"Generated: {datetime.now():%Y-%m-%d %H:%M}", new_x="LMARGIN", new_y="NEXT")
            pdf.cell(0, 8, "GitHub: https://github.com/ffgh-code/insuranceandquantification",
                     new_x="LMARGIN", new_y="NEXT")
            pdf.cell(0, 8, "Demo: https://insuranceandquantification-fbmyvwetabtki2r5m8krac.streamlit.app/",
                     new_x="LMARGIN", new_y="NEXT")
            out_path = f"output/reports/{page.replace(' ', '_')}_{datetime.now():%Y%m%d_%H%M}.pdf"
            import os
            os.makedirs("output/reports", exist_ok=True)
            pdf.output(out_path)
            st.sidebar.success(f"已导出：{out_path}")
        except Exception as e:
            st.sidebar.error(f"导出失败：{str(e)[:80]}")

    with st.spinner("Running full analysis pipeline..."):
        results, pipeline = run_pipeline()

    pages = {
        "Overview": lambda: show_overview(results, pipeline),
        "Market Data": lambda: show_market_data(pipeline),
        "Sentiment Analysis": lambda: show_sentiment(pipeline),
        "Volatility Models": lambda: show_volatility(pipeline),
       "Strategy Backtest": lambda: show_strategy(pipeline),
        "Actuarial Applications": lambda: show_actuarial(pipeline),
        "Rolling Backtest Report": lambda: show_rolling_backtest(pipeline),
        "Regime Performance": lambda: show_regime_performance(pipeline),
        "Solvency Simulation": lambda: show_solvency_simulation(pipeline),
        "Reserving Comparison": lambda: show_reserving_comparison(pipeline),
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
                sc.set_volatility_input(cond_vol)
                scr_df = sc.compute_market_risk_scr()
                if not scr_df.empty:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Current Vol", f"{cond_vol.iloc[-1]:.2%}")
                    c2.metric("Solvency II SCR", f"${scr_df['scr_solvency_ii'].iloc[-1]:,.0f}")
                    c3.metric("C-ROSS SCR", f"${scr_df['scr_cross'].iloc[-1]:,.0f}")
                    stress = sc.stress_test_extreme_sentiment(1.5)
                    st.dataframe(stress, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            vol_input = st.slider("Annualized Vol", 0.05, 0.60, 0.20, 0.01, key="scr_vol")
        with c2:
            cap_input = st.number_input("Exposure ($)", 1_000_000, 1_000_000_000, 100_000_000, step=1_000_000, key="scr_cap")

        var_995 = SolvencyCalculator.calculate_var(vol_input, 0.995)
        var_99 = SolvencyCalculator.calculate_var(vol_input, 0.99)
        cvar_995 = SolvencyCalculator.calculate_cvar(vol_input, 0.995)

        c1, c2, c3 = st.columns(3)
        c1.metric("VaR (99.5%)", f"${var_995 * cap_input:,.0f}")
        c2.metric("VaR (99%)", f"${var_99 * cap_input:,.0f}")
        c3.metric("CVaR (99.5%)", f"${cvar_995 * cap_input:,.0f}")

    with tab2:
        st.subheader("Insurance Loss Reserving with GARCH")
        st.markdown(
            "Applying volatility models to insurance claim data for loss reserving."
        )

        from actuarial.loss_modeling import LossReserving
        lr = LossReserving()
        data = lr.load()
        vol_series = None
        if pipeline._garch_result is not None:
            vol_series = pipeline._garch_result.get("conditional_volatility")
        comp = lr.compare(data, vol_series)
        st.dataframe(comp, use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("Mortality Improvement Rate Forecasting")
        st.markdown(
            "Comparing Lee-Carter, GARCH, and naive mortality forecasting methods."
        )

        from actuarial.mortality import MortalityForecaster
        mf = MortalityForecaster()
        mort_data = mf.load()
        lc, rs = mf.lee_carter(mort_data)
        lcg = mf.lc_garch(mort_data)
        st.dataframe(mf.compare(), use_container_width=True, hide_index=True)
        c1, c2 = st.columns(2)
        c1.metric("Lee-Carter 10yr", f"{lc[-1]:.6f}")
        c2.metric("GARCH-LC 10yr", f"{lcg['fc'][-1]:.6f}")

def show_rolling_backtest(pipeline):
    st.title("Rolling Window Backtest Report")
    st.markdown("240-day window, monthly roll, 7:3 train/test split.")
    from strategy.rolling_backtest import RollingWindowBacktest
    import numpy as np
    if pipeline._prices is not None and not pipeline._prices.empty:
        prices = pipeline._prices["close"]
        signals = pd.Series(np.random.randn(len(prices))*0.3, index=prices.index)
        rbt = RollingWindowBacktest()
        results = rbt.run(signals, prices)
        if not results.empty:
            st.dataframe(results, use_container_width=True)
            c1, c2 = st.columns(2)
            c1.metric("Avg Sharpe", f"{results['sharpe'].mean():.2f}")
            c2.metric("Std Sharpe", f"{results['sharpe'].std():.2f}")
    st.info("Replace synthetic signals with real strategy output for production use.")

def show_regime_performance(pipeline):
    st.title("Regime-Based Strategy Performance")
    st.markdown("Bull / Bear / Range-bound market regime classification and per-regime backtest.")
    from strategy.rolling_backtest import MarketRegimeClassifier
    if pipeline._prices is not None and not pipeline._prices.empty:
        mrc = MarketRegimeClassifier()
        regime = mrc.classify(pipeline._prices["close"])
        counts = regime.value_counts()
        st.bar_chart(counts)
        st.caption("Regime classification: Bull (>8%/60d), Bear (<-5%/60d), Range (otherwise).")
    st.info("A-share Sharpe is structurally lower than US due to: price limits, short-sale restrictions, retail dominance (80%+ volume), and policy-driven regime shifts.")

def show_solvency_simulation(pipeline):
    st.title("Solvency II / C-ROSS Dynamic Simulation")
    st.markdown("GARCH volatility -> SCR automatic calculation + stress test.")
    from actuarial.solvency import SolvencyCalculator
    sc = SolvencyCalculator()
    if pipeline._garch_result is not None:
        cv = pipeline._garch_result.get("conditional_volatility")
        if cv is not None and not cv.empty:
            sc.set_volatility_input(cv)
            scr_df = sc.compute_market_risk_scr()
            st.line_chart(scr_df.set_index("date")[["scr_solvency_ii", "scr_cross"]] if "date" in scr_df.columns else scr_df)
            stress = sc.stress_test_extreme_sentiment(1.5)
            st.dataframe(stress, use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        vol = st.slider("Volatility for manual test", 0.05, 0.60, 0.20, 0.01)
    with col2:
        cap = st.number_input("Capital Base", 1e6, 1e9, 1e8, step=1e6)
    var = SolvencyCalculator.calculate_var(vol, 0.995)
    st.metric("VaR 99.5%", f"${var * cap:,.0f}")

def show_reserving_comparison(pipeline):
    st.title("Loss Reserving: Static vs Dynamic")
    st.markdown("Comparing static 35% provisioning vs volatility-adjusted dynamic reserving.")
    from actuarial.loss_modeling import LossReserving
    lr = LossReserving()
    data = lr.load()
    vol_series = pipeline._garch_result.get("conditional_volatility") if pipeline._garch_result else None
    comp = lr.compare(data, vol_series)
    st.dataframe(comp, use_container_width=True)
    if vol_series is not None:
        diff = lr.diff_chart(vol_series)
        if not diff.empty and "diff" in diff.columns:
            st.subheader("Reserve Difference Over Years")
            st.bar_chart(diff.set_index("year")[["static","dynamic"]])
    st.caption("Dynamic reserving smooths profit volatility by increasing reserves in high-volatility periods and releasing them in low-volatility periods.")


if __name__ == "__main__":
    main()
