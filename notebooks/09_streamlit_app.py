import json
import re
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from phase9_core import (
    get_paths,
    load_inputs,
    build_rag_documents,
    executive_decision_agent,
)

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Insurance Executive Intelligence Platform",
    layout="wide"
)

# =========================================================
# CACHED LOADERS
# =========================================================
@st.cache_data(show_spinner=False)
def load_app_state():
    paths = get_paths()
    payload_path = paths.ph9 / "phase9_exec_payload.json"

    if not payload_path.exists():
        raise FileNotFoundError("Phase 9 payload not found. Run the notebook first.")

    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    bundle = load_inputs(paths)

    segment = pd.DataFrame(payload["segment"])
    top_actions = pd.DataFrame(payload["top_actions"])
    scenario_summary = bundle["scenario_summary"]
    phase8_exec_report = bundle["phase8_exec_report"]

    docs = build_rag_documents(
        phase6_metrics=payload["phase6_metrics"],
        ep_metrics=payload["ep_metrics"],
        governance=payload["governance_checks"],
        scenario_summary=scenario_summary,
        phase8_exec_report=phase8_exec_report,
        segment=segment,
        strategy_df=top_actions,
    )

    return paths, payload, bundle, segment, top_actions, docs


# =========================================================
# LOAD CORE FILES
# =========================================================
try:
    paths, payload, bundle, segment, top_actions, docs = load_app_state()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

scenario_summary = bundle["scenario_summary"]

# =========================================================
# HELPERS
# =========================================================
def classify_capital_status(solvency_ratio: float) -> str:
    if pd.isna(solvency_ratio):
        return "⚪ Unknown"
    if solvency_ratio < 0.5:
        return "🔴 Critical Capital Deficiency"
    if solvency_ratio < 1.0:
        return "🟠 Material Capital Shortfall"
    if solvency_ratio < 1.5:
        return "🟡 Adequate but Below Target"
    return "🟢 Strong Capital Position"


def normalize_text(text: str) -> str:
    return (
        text.lower()
        .replace("capitalised", "capital")
        .replace("capitalized", "capital")
        .replace("solvency", "capital")
        .replace("underwriting", "underwrite")
        .replace("pricing", "price")
    )


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z_]+", text.lower()))


def retrieve_context(query: str, docs: list[dict[str, str]], top_k: int = 3) -> list[dict[str, str]]:
    q = tokenize(query)

    scored = []
    for d in docs:
        score = len(q & tokenize(d["text"]))
        scored.append((score, d))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = [d for score, d in scored[:top_k] if score > 0]

    return results if results else docs[:1]


def recompute_scenario(
    segment_df: pd.DataFrame,
    inflation_shock: float,
    frequency_shock: float,
    fraud_multiplier: float,
    capital_available: float,
    ep_payload: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, float]]:
    df = segment_df.copy()

    df["scenario_freq"] = df["expected_frequency"] * (1 + frequency_shock)
    df["scenario_sev"] = df["mean_severity"] * (1 + inflation_shock)

    motor_mask = df["product_type"].astype(str).str.lower().eq("motor")
    fraud_adj = np.where(motor_mask, fraud_multiplier, 1.0)

    df["scenario_loss"] = df["scenario_freq"] * df["scenario_sev"] * fraud_adj
    df["scenario_loss_bn"] = df["scenario_loss"] / 1e9
    df["scenario_loss_share"] = df["scenario_loss"] / df["scenario_loss"].sum()

    if "earned_premium" in df.columns:
        df["scenario_pressure"] = df["scenario_loss"] / df["earned_premium"].clip(lower=1e-9)
    else:
        df["scenario_pressure"] = df["scenario_loss"] / max(df["scenario_loss"].median(), 1e-9)

    total_loss = float(df["scenario_loss"].sum())

    # Scale off EP mean to keep scenario EP shift coherent with Monte Carlo base
    base_ep_mean = float(ep_payload["mean_loss_bn"]) * 1e9
    scale = total_loss / base_ep_mean if base_ep_mean > 0 else 1.0

    p995_loss = float(ep_payload["loss_1_in_200_bn"]) * 1e9 * scale
    mean_loss = float(ep_payload["mean_loss_bn"]) * 1e9 * scale
    tvar99_loss = float(ep_payload["tvar_99_bn"]) * 1e9 * scale
    solvency_ratio = capital_available / p995_loss if p995_loss > 0 else np.nan

    metrics = {
        "total_loss": total_loss,
        "mean_loss": mean_loss,
        "p995_loss": p995_loss,
        "tvar99_loss": tvar99_loss,
        "solvency_ratio": solvency_ratio,
        "scale": scale,
    }
    return df, metrics


def build_macro_stress_table(
    segment_df: pd.DataFrame,
    capital_available: float,
    fraud_multiplier: float,
    ep_payload: dict[str, Any],
) -> pd.DataFrame:
    rows = []
    for infl in [0.00, 0.05, 0.10, 0.20]:
        _, m = recompute_scenario(
            segment_df=segment_df,
            inflation_shock=infl,
            frequency_shock=infl * 0.30,
            fraud_multiplier=fraud_multiplier * (1 + infl * 0.20),
            capital_available=capital_available,
            ep_payload=ep_payload,
        )
        rows.append({
            "inflation_shock": infl,
            "portfolio_loss": m["total_loss"],
            "mean_loss": m["mean_loss"],
            "p995_loss": m["p995_loss"],
            "tvar99_loss": m["tvar99_loss"],
            "solvency_ratio": m["solvency_ratio"],
            "scale": m["scale"],
        })
    return pd.DataFrame(rows)


def dynamic_board_agent(
    payload: dict[str, Any],
    scenario_metrics: dict[str, float],
    scenario_segment: pd.DataFrame,
) -> str:
    ranked = scenario_segment.sort_values(
        ["scenario_loss", "scenario_pressure"],
        ascending=False
    ).reset_index(drop=True)

    top = ranked.iloc[0]

    solvency = float(scenario_metrics["solvency_ratio"])
    loss_200 = float(scenario_metrics["p995_loss"]) / 1e9

    if solvency < 0.5:
        capital_msg = "The portfolio exhibits a critical capital deficiency requiring immediate executive intervention."
    elif solvency < 1.0:
        capital_msg = "The portfolio is undercapitalised and requires corrective capital and pricing action."
    elif solvency < 1.5:
        capital_msg = "Capital is adequate but below target levels."
    else:
        capital_msg = "Capital position is strong."

    return (
        f"The portfolio is concentrated in {top['product_type']} / {top['channel']}, "
        f"which represents the highest-risk segment under the current scenario. "
        f"Tail risk remains elevated with a 1-in-200 loss of {loss_200:.3f} bn. "
        f"{capital_msg} "
        f"Management should prioritise repricing, underwriting discipline, and capital optimisation."
    )


# =========================================================
# MULTI-AGENT REASONING LAYER
# =========================================================
def risk_agent(question: str, payload: dict[str, Any], scenario_segment: pd.DataFrame) -> dict[str, Any]:
    ranked = scenario_segment.sort_values(
        ["scenario_loss", "scenario_pressure"],
        ascending=False
    ).reset_index(drop=True)
    top = ranked.iloc[0]

    return {
        "agent": "risk_agent",
        "decision": f"Primary portfolio focus should be {top['product_type']} / {top['channel']}",
        "reasoning": [
            "Ranked scenario segments by scenario loss and scenario pressure",
            f"Top scenario segment = {top['product_type']} / {top['channel']}",
            f"Scenario loss share = {top['scenario_loss_share']:.4f}",
        ],
        "evidence": {
            "product_type": top["product_type"],
            "channel": top["channel"],
            "scenario_loss_bn": float(top["scenario_loss_bn"]),
            "scenario_loss_share": float(top["scenario_loss_share"]),
            "scenario_pressure": float(top["scenario_pressure"]),
        },
    }


def capital_agent(question: str, payload: dict[str, Any], scenario_metrics: dict[str, float]) -> dict[str, Any]:
    solvency_ratio = float(scenario_metrics["solvency_ratio"])
    status = classify_capital_status(solvency_ratio)

    if solvency_ratio < 0.5:
        decision = "Severe capital impairment — immediate intervention required"
    elif solvency_ratio < 1.0:
        decision = "Material capital shortfall"
    elif solvency_ratio < 1.5:
        decision = "Capital adequate but below target"
    else:
        decision = "Capital position acceptable"

    return {
        "agent": "capital_agent",
        "decision": decision,
        "reasoning": [
            "Classified as capital adequacy review",
            f"Scenario P99.5 loss = {scenario_metrics['p995_loss'] / 1e9:.3f} bn",
            f"Scenario solvency ratio = {solvency_ratio:.4f}",
            f"Capital status = {status}",
        ],
        "evidence": {
            "capital_available": payload["capital_metrics"]["capital_available"],
            "scenario_p995_loss": scenario_metrics["p995_loss"],
            "scenario_tvar99_loss": scenario_metrics["tvar99_loss"],
            "scenario_solvency_ratio": solvency_ratio,
            "capital_status": status,
        },
    }


def strategy_agent(question: str, payload: dict[str, Any], scenario_segment: pd.DataFrame) -> dict[str, Any]:
    df = scenario_segment.copy().sort_values(
        ["scenario_pressure", "scenario_loss_share"],
        ascending=False
    ).reset_index(drop=True)
    top = df.iloc[0]
    base_metric = float(top["scenario_pressure"])

    if base_metric > 0.13:
        action_label = "Reprice aggressively and tighten underwriting"
        rate_change = 0.08
    elif base_metric > 0.115:
        action_label = "Reprice moderately and review underwriting"
        rate_change = 0.05
    else:
        action_label = "Targeted underwriting review"
        rate_change = 0.03

    return {
        "agent": "strategy_agent",
        "decision": f"{action_label} in {top['product_type']} / {top['channel']}",
        "reasoning": [
            "Classified as pricing / underwriting decision",
            f"Top scenario pressure segment = {top['product_type']} / {top['channel']}",
            f"Scenario pressure metric = {base_metric:.4f}",
            f"Indicative rate action = {rate_change:.2%}",
        ],
        "evidence": {
            "product_type": top["product_type"],
            "channel": top["channel"],
            "scenario_pressure": base_metric,
            "scenario_loss_bn": float(top["scenario_loss_bn"]),
            "recommended_rate_change_pct": rate_change,
        },
    }


def governance_agent(question: str, payload: dict[str, Any]) -> dict[str, Any]:
    gov = payload["governance_checks"]
    passed = payload["governance_ok"]

    return {
        "agent": "governance_agent",
        "decision": "Governance controls are passed" if passed else "Governance controls are not fully passed",
        "reasoning": [
            "Classified as governance / control review",
            f"Exposure valid = {gov.get('exposure_valid')}",
            f"Frequency valid = {gov.get('frequency_valid')}",
            f"Fraud layer present = {gov.get('fraud_layer_present')}",
            f"Scenario defined = {gov.get('scenario_defined')}",
        ],
        "evidence": gov,
    }


def scenario_agent(
    question: str,
    payload: dict[str, Any],
    macro_results_df: pd.DataFrame,
) -> dict[str, Any]:
    worst = macro_results_df.sort_values("solvency_ratio", ascending=True).iloc[0]

    return {
        "agent": "scenario_agent",
        "decision": "Maximum capital strain occurs under extreme inflation stress",
        "reasoning": [
            "Classified as scenario stress analysis",
            f"Lowest solvency observed = {worst['solvency_ratio']:.4f}",
            f"Inflation shock at worst case = {worst['inflation_shock'] * 100:.0f}%",
            f"Worst-case P99.5 loss = {worst['p995_loss'] / 1e9:.3f} bn",
        ],
        "evidence": worst.to_dict(),
    }


def router_agent(question: str) -> str:
    q = normalize_text(question)

    if any(k in q for k in ["scenario", "stress", "maximum", "worst", "strain"]):
        return "scenario"
    if any(k in q for k in ["capital", "solvency", "scr"]):
        return "capital"
    if any(k in q for k in ["price", "underwrite", "action", "repric"]):
        return "strategy"
    if any(k in q for k in ["governance", "control", "assumption", "trust"]):
        return "governance"
    if any(k in q for k in ["risk", "focus", "priority", "segment", "concentration"]):
        return "risk"
    return "executive_fallback"


def multi_agent_reasoner(
    question: str,
    payload: dict[str, Any],
    scenario_segment: pd.DataFrame,
    scenario_metrics: dict[str, float],
    macro_results_df: pd.DataFrame,
    docs: list[dict[str, str]],
) -> dict[str, Any]:
    route = router_agent(question)

    if route == "scenario":
        result = scenario_agent(question, payload, macro_results_df)
    elif route == "capital":
        result = capital_agent(question, payload, scenario_metrics)
    elif route == "strategy":
        result = strategy_agent(question, payload, scenario_segment)
    elif route == "governance":
        result = governance_agent(question, payload)
    elif route == "risk":
        result = risk_agent(question, payload, scenario_segment)
    else:
        result = executive_decision_agent(question, payload)
        result["agent"] = "executive_fallback"

    result["retrieved_context"] = retrieve_context(question, docs, top_k=3)
    return result


# =========================================================
# BASE VALUES
# =========================================================
base_capital_available = float(payload["capital_metrics"]["capital_available"])
base_solvency = float(payload["capital_metrics"]["solvency_ratio"])
base_status = classify_capital_status(base_solvency)

default_inflation = (
    float(scenario_summary["inflation_shock"].iloc[0])
    if "inflation_shock" in scenario_summary.columns
    else 0.12
)
default_frequency = (
    float(scenario_summary["frequency_shock"].iloc[0])
    if "frequency_shock" in scenario_summary.columns
    else 0.08
)
default_fraud = (
    float(scenario_summary["fraud_multiplier"].iloc[0])
    if "fraud_multiplier" in scenario_summary.columns
    else 1.25
)

# =========================================================
# SIDEBAR — INTERACTIVE SCENARIO ENGINE
# =========================================================
st.sidebar.header("Interactive Scenario Engine")

inflation_input = st.sidebar.slider(
    "Inflation Shock",
    min_value=0.00,
    max_value=0.30,
    value=float(default_inflation),
    step=0.01,
    format="%.2f",
)

frequency_input = st.sidebar.slider(
    "Frequency Shock",
    min_value=0.00,
    max_value=0.30,
    value=float(default_frequency),
    step=0.01,
    format="%.2f",
)

fraud_input = st.sidebar.slider(
    "Motor Fraud Multiplier",
    min_value=1.00,
    max_value=2.00,
    value=float(default_fraud),
    step=0.05,
    format="%.2f",
)

capital_input = st.sidebar.number_input(
    "Available Capital",
    min_value=0.0,
    value=float(base_capital_available),
    step=10_000_000.0,
)

scenario_segment, scenario_metrics = recompute_scenario(
    segment_df=segment,
    inflation_shock=inflation_input,
    frequency_shock=frequency_input,
    fraud_multiplier=fraud_input,
    capital_available=capital_input,
    ep_payload=payload["ep_metrics"],
)

scenario_status = classify_capital_status(scenario_metrics["solvency_ratio"])

macro_results_df = build_macro_stress_table(
    segment_df=segment,
    capital_available=capital_input,
    fraud_multiplier=fraud_input,
    ep_payload=payload["ep_metrics"],
)

# =========================================================
# HEADER
# =========================================================
st.title("Insurance Digital Twin — Executive Intelligence Cockpit")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Policies", f"{payload['snapshot']['policies']:,}")
m2.metric("Claims", f"{payload['snapshot']['claims']:,}")
m3.metric("Paid", f"{payload['snapshot']['paid']:,.0f}")
m4.metric("Reserve", f"{payload['snapshot']['reserve']:,.0f}")
m5.metric("Base Solvency", f"{base_solvency:.3f}", delta=base_status)

s1, s2, s3, s4 = st.columns(4)
s1.metric("Scenario Loss (£m)", f"{scenario_metrics['total_loss']/1e6:,.1f}")
s2.metric("Scenario P99.5 (£bn)", f"{scenario_metrics['p995_loss']/1e9:,.3f}")
s3.metric("Scenario TVaR99 (£bn)", f"{scenario_metrics['tvar99_loss']/1e9:,.3f}")
s4.metric("Scenario Solvency", f"{scenario_metrics['solvency_ratio']:.3f}", delta=scenario_status)

st.markdown("---")

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Executive Summary",
    "Strategic Actions",
    "Risk Concentration",
    "Executive AI",
    "Scenario Engine",
])

# =========================================================
# TAB 1 — EXECUTIVE SUMMARY
# =========================================================
with tab1:
    st.subheader("Board Summary")
    board_text = dynamic_board_agent(payload, scenario_metrics, scenario_segment)
    st.write(board_text)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Tail Risk Metrics")
        st.json(payload["ep_metrics"])
    with col2:
        st.markdown("#### Capital Metrics")
        st.json(payload["capital_metrics"])

    st.markdown("#### Scenario Commentary")
    st.info(
        f"Under the current interactive scenario, solvency moves to "
        f"{scenario_metrics['solvency_ratio']:.4f} ({scenario_status})."
    )

# =========================================================
# TAB 2 — STRATEGIC ACTIONS
# =========================================================
with tab2:
    st.subheader("Priority Actions")

    if not top_actions.empty:
        top = top_actions.iloc[0]
        st.success(
            f"Top Priority: {top['action']} in {top['product_type']} / {top['channel']}"
        )

    st.dataframe(top_actions, use_container_width=True)

    fig_actions = px.scatter(
        top_actions,
        x="metric_before",
        y="metric_after",
        size="capital_share_proxy",
        color="action",
        hover_data=["product_type", "channel", "rationale"],
        title=f"Strategy Advisor — {payload['pressure_label']} Before vs After"
    )
    st.plotly_chart(fig_actions, use_container_width=True)

# =========================================================
# TAB 3 — RISK CONCENTRATION
# =========================================================
with tab3:
    st.subheader("Portfolio Risk Structure")

    fig_tree = px.treemap(
        scenario_segment,
        path=[px.Constant("Portfolio"), "product_type", "channel"],
        values="scenario_loss",
        color="scenario_pressure",
        color_continuous_scale="Reds",
        title="Scenario Risk Concentration"
    )
    st.plotly_chart(fig_tree, use_container_width=True)

    fig_bar = px.bar(
        scenario_segment.sort_values("scenario_loss", ascending=False).head(12),
        x="product_type",
        y="scenario_loss_bn",
        color="channel",
        barmode="group",
        title="Top Scenario Risk Drivers (£bn)"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# =========================================================
# TAB 4 — EXECUTIVE AI (MULTI-AGENT + RAG)
# =========================================================
with tab4:
    st.subheader("Executive Decision AI — Multi-Agent Reasoning")

    suggestions = [
        "Is the portfolio adequately capitalised?",
        "Where should leadership focus first?",
        "What pricing action should we take?",
        "Is governance acceptable?",
        "What scenario causes maximum capital strain?",
        "Which segment drives the most risk under this scenario?",
    ]

    selected_q = st.selectbox("Suggested questions", [""] + suggestions)
    user_q = st.text_input("Or ask your own question")
    q = user_q if user_q else selected_q

    if q:
        result = multi_agent_reasoner(
            question=q,
            payload=payload,
            scenario_segment=scenario_segment,
            scenario_metrics=scenario_metrics,
            macro_results_df=macro_results_df,
            docs=docs,
        )

        st.markdown("### Routed Agent")
        st.code(result["agent"])

        st.markdown("### Decision")
        st.success(result["decision"])

        st.markdown("### Reasoning")
        for r in result["reasoning"]:
            st.write(f"- {r}")

        st.markdown("### Evidence")
        st.json(result["evidence"])

        with st.expander("📚 Supporting Context (RAG)"):
            seen_sources = set()
            for d in result["retrieved_context"]:
                source = d.get("source", "unknown")
                if source in seen_sources:
                    continue
                seen_sources.add(source)
                st.markdown(f"**[{source}]**")
                st.text(d["text"][:800])

# =========================================================
# TAB 5 — INTERACTIVE SCENARIO ENGINE
# =========================================================
with tab5:
    st.subheader("Scenario Impact on Solvency and Tail Risk")

    fig_sol = go.Figure()
    fig_sol.add_trace(go.Scatter(
        x=macro_results_df["inflation_shock"] * 100,
        y=macro_results_df["solvency_ratio"],
        mode="lines+markers+text",
        text=[f"{v:.3f}" for v in macro_results_df["solvency_ratio"]],
        textposition="top center",
        name="Solvency Ratio"
    ))
    fig_sol.add_hline(y=1.0, line_dash="dash", annotation_text="Regulatory Minimum (1.0)")
    fig_sol.add_hline(y=1.5, line_dash="dot", annotation_text="Comfort Zone (1.5)")
    fig_sol.update_layout(
        title="Inflation Shock vs Solvency Ratio",
        xaxis_title="Inflation Shock (%)",
        yaxis_title="Solvency Ratio",
        yaxis=dict(range=[0, 2.5]),
        height=500
    )
    st.plotly_chart(fig_sol, use_container_width=True)

    fig_tail = go.Figure()
    xvals = (macro_results_df["inflation_shock"] * 100).astype(int).astype(str) + "%"

    fig_tail.add_trace(go.Bar(
        x=xvals,
        y=macro_results_df["p995_loss"] / 1e9,
        name="P99.5 Loss (SCR proxy)"
    ))

    fig_tail.add_trace(go.Scatter(
        x=xvals,
        y=macro_results_df["solvency_ratio"],
        mode="lines+markers+text",
        text=[f"{v:.3f}" for v in macro_results_df["solvency_ratio"]],
        textposition="top center",
        name="Solvency Ratio",
        yaxis="y2"
    ))

    fig_tail.update_layout(
        title="Tail Risk and Solvency Under Macro Stress",
        xaxis_title="Inflation Shock",
        yaxis=dict(title="P99.5 Loss (£bn) — Solvency Capital Requirement (SCR proxy)"),
        yaxis2=dict(
            title="Solvency Ratio (Eligible Capital / SCR)",
            overlaying="y",
            side="right",
            range=[0, 2.5]
        ),
        height=550
    )
    st.plotly_chart(fig_tail, use_container_width=True)

    st.markdown("### Scenario Segment View")
    display_cols = [
        "product_type",
        "channel",
        "scenario_loss_bn",
        "scenario_loss_share",
        "scenario_pressure",
        "mean_severity",
        "expected_frequency",
    ]
    st.dataframe(
        scenario_segment.sort_values("scenario_loss", ascending=False)[display_cols],
        use_container_width=True
    )