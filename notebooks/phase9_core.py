from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import re
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class Paths:
    root: Path
    data: Path
    ph6: Path
    ph7: Path
    ph8: Path
    ph9: Path


def detect_root() -> Path:
    candidates = [
        Path.cwd(),
        Path.cwd().parent,
        Path(r"V:\Himika\Project\insurance-digital-twin"),
    ]
    for candidate in candidates:
        if (candidate / "data" / "raw" / "claims.csv").exists():
            return candidate
    raise FileNotFoundError("Could not detect project root containing data/raw/claims.csv")


def get_paths(root: Path | None = None) -> Paths:
    root = root or detect_root()
    paths = Paths(
        root=root,
        data=root / "data" / "raw",
        ph6=root / "notebooks" / "outputs" / "phase6",
        ph7=root / "notebooks" / "outputs" / "phase7",
        ph8=root / "notebooks" / "outputs" / "phase8",
        ph9=root / "notebooks" / "outputs" / "phase9",
    )
    paths.ph9.mkdir(parents=True, exist_ok=True)
    return paths


def find_col(df: pd.DataFrame, candidates: list[str], default: str | None = None) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return default


def detect_loss_column(df: pd.DataFrame) -> str:
    candidates = [
        "portfolio_loss",
        "loss",
        "total_loss",
        "simulated_loss",
        "aggregate_loss",
        "portfolio_total_loss",
        "loss_distribution",
    ]
    for c in candidates:
        if c in df.columns:
            return c

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        return numeric_cols[0]

    raise ValueError("No numeric loss column found in loss distribution file")


def load_inputs(paths: Paths) -> dict[str, Any]:
    required_files = [
        paths.data / "claims.csv",
        paths.data / "policies.csv",
        paths.data / "policyholders.csv",
        paths.data / "macro.csv",
        paths.ph6 / "phase6_exec_metrics.json",
        paths.ph6 / "relativities_product.csv",
        paths.ph6 / "relativities_channel.csv",
        paths.ph7 / "phase7_siu_capacity_table.csv",
        paths.ph7 / "phase7_siu_cost_capacity_table.csv",
        paths.ph7 / "phase7_siu_threshold_policy_table.csv",
        paths.ph8 / "phase8_ep_metrics.json",
        paths.ph8 / "phase8_governance_checks.json",
        paths.ph8 / "phase8_exec_report.txt",
        paths.ph8 / "scenario_summary.csv",
        paths.ph8 / "loss_distribution_mc.csv",
    ]
    missing = [str(f) for f in required_files if not f.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    claims = pd.read_csv(paths.data / "claims.csv")
    policies = pd.read_csv(paths.data / "policies.csv")
    policyholders = pd.read_csv(paths.data / "policyholders.csv")
    macro = pd.read_csv(paths.data / "macro.csv")

    rel_prod = pd.read_csv(paths.ph6 / "relativities_product.csv")
    rel_channel = pd.read_csv(paths.ph6 / "relativities_channel.csv")

    siu_capacity = pd.read_csv(paths.ph7 / "phase7_siu_capacity_table.csv")
    siu_cost = pd.read_csv(paths.ph7 / "phase7_siu_cost_capacity_table.csv")
    siu_threshold = pd.read_csv(paths.ph7 / "phase7_siu_threshold_policy_table.csv")

    scenario_summary = pd.read_csv(paths.ph8 / "scenario_summary.csv")
    loss_distribution = pd.read_csv(paths.ph8 / "loss_distribution_mc.csv")
    loss_col = detect_loss_column(loss_distribution)
    if loss_col != "portfolio_loss":
        loss_distribution = loss_distribution.rename(columns={loss_col: "portfolio_loss"})

    with open(paths.ph6 / "phase6_exec_metrics.json", "r", encoding="utf-8") as f:
        phase6_metrics = json.load(f)

    with open(paths.ph8 / "phase8_ep_metrics.json", "r", encoding="utf-8") as f:
        ep_metrics = json.load(f)

    with open(paths.ph8 / "phase8_governance_checks.json", "r", encoding="utf-8") as f:
        governance = json.load(f)

    phase8_exec_report = (paths.ph8 / "phase8_exec_report.txt").read_text(encoding="utf-8")

    return {
        "claims": claims,
        "policies": policies,
        "policyholders": policyholders,
        "macro": macro,
        "rel_prod": rel_prod,
        "rel_channel": rel_channel,
        "siu_capacity": siu_capacity,
        "siu_cost": siu_cost,
        "siu_threshold": siu_threshold,
        "scenario_summary": scenario_summary,
        "loss_distribution": loss_distribution,
        "phase6_metrics": phase6_metrics,
        "ep_metrics": ep_metrics,
        "governance": governance,
        "phase8_exec_report": phase8_exec_report,
    }


def prepare_portfolio(
    claims: pd.DataFrame,
    policies: pd.DataFrame,
    rel_prod: pd.DataFrame,
    rel_channel: pd.DataFrame,
) -> dict[str, Any]:
    claims = claims.copy()
    policies = policies.copy()

    paid_col = find_col(claims, ["paid_amount", "paid_loss", "claim_paid", "paid"])
    reserve_col = find_col(claims, ["outstanding_reserve", "reserve", "case_reserve", "os_reserve"])
    fraud_col = find_col(claims, ["is_fraud", "fraud_flag", "suspected_fraud"])
    claim_id_col = find_col(claims, ["claim_id", "claim_number", "id_claim"], default="claim_id")

    product_col = find_col(policies, ["product_type", "product", "lob", "line_of_business"], default="product_type")
    channel_col = find_col(policies, ["channel", "distribution_channel", "sales_channel"], default="channel")
    premium_col = find_col(policies, ["base_annual_premium", "annual_premium", "written_premium", "premium", "gross_premium"])
    start_col = find_col(policies, ["start_date", "policy_start_date", "inception_date"])
    end_col = find_col(policies, ["end_date", "policy_end_date", "expiry_date"])

    if paid_col is None or reserve_col is None:
        raise ValueError("Required claims columns for paid/reserve were not found.")

    claims[paid_col] = pd.to_numeric(claims[paid_col], errors="coerce").fillna(0)
    claims[reserve_col] = pd.to_numeric(claims[reserve_col], errors="coerce").fillna(0)
    claims["severity"] = claims[paid_col].clip(lower=0) + claims[reserve_col].clip(lower=0)

    if start_col and end_col:
        policies[start_col] = pd.to_datetime(policies[start_col], errors="coerce")
        policies[end_col] = pd.to_datetime(policies[end_col], errors="coerce")
        policies["exposure"] = (
            ((policies[end_col] - policies[start_col]).dt.days / 365.25)
            .clip(lower=0, upper=1)
            .fillna(1.0)
        )
    else:
        policies["exposure"] = 1.0

    prod_map = dict(zip(rel_prod["level"], rel_prod["relativity"]))
    channel_map = dict(zip(rel_channel["level"], rel_channel["relativity"]))

    policies["relativity_prod"] = policies[product_col].map(prod_map).fillna(1.0)
    policies["relativity_channel"] = policies[channel_col].map(channel_map).fillna(1.0)

    if premium_col is not None:
        policies[premium_col] = pd.to_numeric(policies[premium_col], errors="coerce").fillna(0)
        policies["earned_premium"] = policies[premium_col] * policies["exposure"]
        premium_basis = "actual_or_exported_premium"
    else:
        policies["earned_premium"] = (
            1000
            * policies["relativity_prod"]
            * policies["relativity_channel"]
            * policies["exposure"]
        )
        premium_basis = "synthetic_premium_proxy"

    severity_by_segment = (
        claims.merge(
            policies[["policy_id", product_col, channel_col]],
            on="policy_id",
            how="left"
        )
        .groupby([product_col, channel_col], dropna=False)["severity"]
        .mean()
        .reset_index()
        .rename(columns={"severity": "mean_severity"})
    )

    policies = policies.merge(severity_by_segment, on=[product_col, channel_col], how="left")
    policies["mean_severity"] = policies["mean_severity"].fillna(claims["severity"].mean())

    merged = claims.merge(policies, on="policy_id", how="left", suffixes=("_claim", "_policy"))

    return {
        "claims": claims,
        "policies": policies,
        "merged": merged,
        "paid_col": paid_col,
        "reserve_col": reserve_col,
        "fraud_col": fraud_col,
        "claim_id_col": claim_id_col,
        "product_col": product_col,
        "channel_col": channel_col,
        "premium_basis": premium_basis,
    }


def build_segment_view(
    policies: pd.DataFrame,
    merged: pd.DataFrame,
    product_col: str,
    channel_col: str,
    claim_id_col: str,
    fraud_multiplier: float,
    inflation_shock: float,
    frequency_shock: float,
    premium_basis: str,
) -> pd.DataFrame:
    portfolio = policies.copy()

    portfolio["expected_frequency"] = (
        0.05 * portfolio["relativity_prod"] * portfolio["relativity_channel"] * portfolio["exposure"]
    )
    portfolio["freq_stressed"] = portfolio["expected_frequency"] * (1 + frequency_shock)
    portfolio["sev_stressed"] = portfolio["mean_severity"] * (1 + inflation_shock)

    portfolio["stressed_loss"] = portfolio["freq_stressed"] * portfolio["sev_stressed"]
    portfolio["stressed_loss"] = np.where(
        portfolio[product_col].astype(str).str.lower().eq("motor"),
        portfolio["stressed_loss"] * fraud_multiplier,
        portfolio["stressed_loss"]
    )

    segment = (
        portfolio.groupby([product_col, channel_col], as_index=False)
        .agg(
            policies=("policy_id", "count"),
            earned_premium=("earned_premium", "sum"),
            expected_frequency=("expected_frequency", "sum"),
            freq_stressed=("freq_stressed", "sum"),
            stressed_loss=("stressed_loss", "sum"),
            mean_severity=("mean_severity", "mean"),
        )
    )

    actual_claims = (
        merged.groupby([product_col, channel_col], dropna=False)
        .agg(
            actual_claim_count=(claim_id_col, "count"),
            actual_incurred=("severity", "sum"),
            actual_avg_severity=("severity", "mean"),
        )
        .reset_index()
    )

    segment = segment.merge(actual_claims, on=[product_col, channel_col], how="left")
    segment["loss_share"] = segment["stressed_loss"] / segment["stressed_loss"].sum()
    segment["capital_share_proxy"] = segment["loss_share"]
    segment["stressed_loss_bn"] = segment["stressed_loss"] / 1e9

    if premium_basis == "actual_or_exported_premium":
        segment["loss_ratio_stressed"] = segment["stressed_loss"] / segment["earned_premium"].clip(lower=1e-9)
        segment["pressure_metric"] = segment["loss_ratio_stressed"]
        segment["pressure_label"] = "loss_ratio_stressed"
    else:
        segment["incurred_per_policy"] = segment["stressed_loss"] / segment["policies"].clip(lower=1)
        median_val = segment["incurred_per_policy"].median()
        segment["pressure_metric"] = segment["incurred_per_policy"] / max(median_val, 1e-9)
        segment["pressure_label"] = "relative_pressure_index"

    return segment.sort_values("stressed_loss", ascending=False).reset_index(drop=True)


def governance_pass(governance: dict[str, Any]) -> bool:
    return (
        governance.get("exposure_valid", False)
        and governance.get("frequency_valid", False)
        and governance.get("fraud_layer_present", False)
        and governance.get("scenario_defined", False)
        and governance.get("relativity_prod_missing", 1) == 0
        and governance.get("relativity_channel_missing", 1) == 0
    )


def build_snapshot(
    claims: pd.DataFrame,
    policies: pd.DataFrame,
    paid_col: str,
    reserve_col: str,
    fraud_col: str | None,
) -> dict[str, float]:
    return {
        "policies": int(len(policies)),
        "claims": int(len(claims)),
        "paid": float(claims[paid_col].sum()),
        "reserve": float(claims[reserve_col].sum()),
        "avg_severity": float(claims["severity"].mean()),
        "fraud_rate": float(claims[fraud_col].mean()) if fraud_col else np.nan,
    }


def get_capital_metrics(
    scenario_summary: pd.DataFrame,
    ep_metrics: dict[str, Any],
) -> dict[str, float]:
    if "capital_required" in scenario_summary.columns:
        capital_required = float(scenario_summary["capital_required"].iloc[0])
    elif "required_capital" in scenario_summary.columns:
        capital_required = float(scenario_summary["required_capital"].iloc[0])
    else:
        capital_required = float(ep_metrics["loss_1_in_200_bn"]) * 1e9

    if "capital_available" in scenario_summary.columns:
        capital_available = float(scenario_summary["capital_available"].iloc[0])
    else:
        capital_available = np.nan

    solvency_ratio = capital_available / capital_required if capital_required else np.nan

    return {
        "capital_required": capital_required,
        "capital_available": capital_available,
        "solvency_ratio": solvency_ratio,
    }


def build_rag_documents(
    phase6_metrics: dict[str, Any],
    ep_metrics: dict[str, Any],
    governance: dict[str, Any],
    scenario_summary: pd.DataFrame,
    phase8_exec_report: str,
    segment: pd.DataFrame,
    strategy_df: pd.DataFrame,
) -> list[dict[str, str]]:
    return [
        {"source": "phase6_exec_metrics", "text": json.dumps(phase6_metrics, indent=2)},
        {"source": "phase8_ep_metrics", "text": json.dumps(ep_metrics, indent=2)},
        {"source": "phase8_governance", "text": json.dumps(governance, indent=2)},
        {"source": "phase8_exec_report", "text": phase8_exec_report},
        {"source": "scenario_summary", "text": scenario_summary.to_string(index=False)},
        {"source": "segment_top10", "text": segment.head(10).to_string(index=False)},
        {"source": "strategy_top10", "text": strategy_df.head(10).to_string(index=False)},
    ]


def normalize_text(text: str) -> str:
    text = text.lower()
    replacements = {
        "capitalised": "capital",
        "capitalized": "capital",
        "solvency": "capital",
        "fraudulent": "fraud",
        "pricing": "price",
        "underwriting": "underwrite",
        "underpriced": "price",
        "underpricing": "price",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z_]+", normalize_text(text)))


def retrieve_context(query: str, docs: list[dict[str, str]], top_k: int = 3) -> list[dict[str, str]]:
    q = tokenize(query)
    scored = []
    for d in docs:
        score = len(q & tokenize(d["text"]))
        scored.append((score, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    results = [d for score, d in scored[:top_k] if score > 0]
    return results if results else docs[:1]


def build_strategy(segment: pd.DataFrame, premium_basis: str) -> pd.DataFrame:
    elasticity_map = {
        "motor": -1.2,
        "home": -0.8,
        "health": -0.6,
        "warranty": -0.9,
    }

    df = segment.copy()
    df["elasticity"] = df["product_type"].map(elasticity_map).fillna(-1.0)

    recommendations = []
    for _, row in df.iterrows():
        action = "Maintain current stance"
        rate_change = 0.0
        rationale = []

        metric_value = row["pressure_metric"]

        if premium_basis == "actual_or_exported_premium":
            if metric_value > 0.13:
                action = "Increase pricing aggressively"
                rate_change = 0.08
                rationale.append("Stressed loss ratio materially above portfolio tolerance")
            elif metric_value > 0.115:
                action = "Increase pricing moderately"
                rate_change = 0.05
                rationale.append("Stressed loss ratio above tolerance")
            elif metric_value > 0.105:
                action = "Review pricing and underwriting"
                rate_change = 0.03
                rationale.append("Stressed loss ratio slightly elevated")
            else:
                rationale.append("Stressed loss ratio within acceptable range")
        else:
            if metric_value > 1.30:
                action = "Tighten underwriting and reprice selectively"
                rate_change = 0.05
                rationale.append("Relative pressure materially above segment median")
            elif metric_value > 1.10:
                action = "Review pricing and underwriting"
                rate_change = 0.03
                rationale.append("Relative pressure above peer group")
            else:
                rationale.append("Relative pressure within normal range")

        if row["capital_share_proxy"] > 0.20:
            rationale.append("High contributor to stressed portfolio losses")
        if str(row["product_type"]).lower() == "motor":
            rationale.append("Fraud-overlay sensitive line")

        retention_factor = max(0.5, min(1.05, 1 + row["elasticity"] * rate_change))

        recommendations.append({
            "product_type": row["product_type"],
            "channel": row["channel"],
            "action": action,
            "recommended_rate_change_pct": rate_change,
            "retention_factor": retention_factor,
            "metric_before": metric_value,
            "metric_after": metric_value * (1 - rate_change) if premium_basis != "actual_or_exported_premium" else metric_value / max((1 + rate_change) * retention_factor, 1e-9),
            "capital_share_proxy": row["capital_share_proxy"],
            "rationale": "; ".join(rationale),
        })

    return pd.DataFrame(recommendations).sort_values(
        ["metric_before", "capital_share_proxy"], ascending=[False, False]
    ).reset_index(drop=True)


def rank_segments_for_decision(segment: pd.DataFrame) -> pd.DataFrame:
    df = segment.copy()

    if "loss_ratio_stressed" in df.columns:
        df["risk_score"] = (
            0.45 * (df["loss_ratio_stressed"] / df["loss_ratio_stressed"].max())
            + 0.35 * (df["loss_share"] / df["loss_share"].max())
            + 0.20 * (df["mean_severity"] / df["mean_severity"].max())
        )
    else:
        df["risk_score"] = (
            0.50 * (df["pressure_metric"] / df["pressure_metric"].max())
            + 0.30 * (df["loss_share"] / df["loss_share"].max())
            + 0.20 * (df["mean_severity"] / df["mean_severity"].max())
        )

    return df.sort_values("risk_score", ascending=False).reset_index(drop=True)


def simulate_segment_action(
    segment_row: pd.Series,
    action_type: str,
    rate_change: float = 0.0,
    severity_reduction: float = 0.0,
    frequency_reduction: float = 0.0,
) -> dict[str, float]:
    base_loss = float(segment_row["stressed_loss"])
    base_premium = float(segment_row["earned_premium"])
    base_metric = float(segment_row["pressure_metric"])

    new_loss = base_loss * (1 - severity_reduction) * (1 - frequency_reduction)
    new_premium = base_premium * (1 + rate_change)

    if new_premium > 0:
        new_metric = new_loss / new_premium
    else:
        new_metric = base_metric

    return {
        "action_type": action_type,
        "base_loss": base_loss,
        "new_loss": new_loss,
        "base_metric": base_metric,
        "new_metric": new_metric,
        "loss_improvement": base_loss - new_loss,
        "premium_change": new_premium - base_premium,
    }


def executive_decision_agent(question: str, payload: dict[str, Any]) -> dict[str, Any]:
    q = normalize_text(question)

    segment = pd.DataFrame(payload["segment"])
    ranked = rank_segments_for_decision(segment)
    top = ranked.iloc[0]

    cap = payload["capital_metrics"]
    ep = payload["ep_metrics"]
    gov = payload["governance_checks"]

    reasoning: list[str] = []
    decision = ""
    evidence: dict[str, Any] = {}

    if "capital" in q or "solvency" in q:
        reasoning.append("Question classified as capital adequacy")
        reasoning.append(f"Capital required = {cap['capital_required']:,.0f}")
        reasoning.append(f"Capital available = {cap['capital_available']:,.0f}")
        reasoning.append(f"Solvency ratio = {cap['solvency_ratio']:.4f}")

        if cap["solvency_ratio"] < 0.10:
            decision = "Critical capital deficiency"
        elif cap["solvency_ratio"] < 0.50:
            decision = "Portfolio is undercapitalised"
        else:
            decision = "Capital position is acceptable"

        evidence = {
            "capital_required": cap["capital_required"],
            "capital_available": cap["capital_available"],
            "solvency_ratio": cap["solvency_ratio"],
            "loss_1_in_200_bn": ep["loss_1_in_200_bn"],
            "tvar_99_bn": ep["tvar_99_bn"],
        }

    elif "risk" in q or "priority" in q or "focus" in q:
        reasoning.append("Question classified as risk prioritisation")
        reasoning.append("Segments ranked using composite risk score")
        reasoning.append(f"Top ranked segment = {top['product_type']} / {top['channel']}")

        decision = f"Focus on {top['product_type']} / {top['channel']}"
        evidence = top.to_dict()

    elif "price" in q or "underwrite" in q or "action" in q:
        reasoning.append("Question classified as pricing / underwriting action")
        reasoning.append(f"Highest-risk segment = {top['product_type']} / {top['channel']}")

        base_metric = float(top["pressure_metric"])

        if base_metric > 0.13:
            chosen_rate = 0.08
            sev_red = 0.04
            freq_red = 0.03
            action_label = "Reprice aggressively and tighten underwriting"
        elif base_metric > 0.115:
            chosen_rate = 0.05
            sev_red = 0.03
            freq_red = 0.02
            action_label = "Reprice moderately and review underwriting"
        else:
            chosen_rate = 0.03
            sev_red = 0.01
            freq_red = 0.01
            action_label = "Targeted underwriting review"

        simulation = simulate_segment_action(
            top,
            action_type="repricing_and_underwriting",
            rate_change=chosen_rate,
            severity_reduction=sev_red,
            frequency_reduction=freq_red,
        )

        decision = f"{action_label} in {top['product_type']} / {top['channel']}"
        evidence = {
            "segment": f"{top['product_type']} / {top['channel']}",
            "simulation": simulation,
            "risk_score": top["risk_score"],
            "base_metric": base_metric,
        }

    elif "governance" in q or "control" in q:
        reasoning.append("Question classified as governance")
        decision = "Governance controls are passed" if payload["governance_ok"] else "Governance controls not fully passed"
        evidence = gov

    else:
        reasoning.append("Question routed to generic executive interpretation")
        decision = payload["board_summary"]
        evidence = {"summary": payload["board_summary"]}

    return {
        "decision": decision,
        "reasoning": reasoning,
        "evidence": evidence,
    }


def classify_capital_severity(solvency_ratio: float) -> str:
    if solvency_ratio < 0.5:
        return "critical capital deficiency"
    elif solvency_ratio < 1.0:
        return "material capital shortfall"
    elif solvency_ratio < 1.5:
        return "adequate but below target"
    else:
        return "strong capital position"


def board_commentary_agent(payload: dict[str, Any]) -> str:
    segment = pd.DataFrame(payload["segment"])
    ranked = rank_segments_for_decision(segment)
    top = ranked.iloc[0]

    cap = payload["capital_metrics"]
    ep = payload["ep_metrics"]

    solvency = cap["solvency_ratio"]
    severity = classify_capital_severity(solvency)

    return (
        f"The portfolio is concentrated in {top['product_type']} / {top['channel']}, "
        f"which ranks highest under the composite risk framework. "
        f"Tail risk remains elevated, with a 1-in-200 loss of {ep['loss_1_in_200_bn']:.4f} bn. "
        f"The portfolio exhibits a {severity}, with a solvency ratio of {solvency:.4f}. "
        f"Immediate management action is required across pricing, underwriting discipline, and capital optimisation."
    )


def build_payload(
    snapshot: dict[str, Any],
    premium_basis: str,
    pressure_label: str,
    phase6_metrics: dict[str, Any],
    ep_metrics: dict[str, Any],
    governance: dict[str, Any],
    governance_ok: bool,
    capital_required: float,
    capital_available: float,
    solvency_ratio: float,
    risk_analysis: str,
    capital_analysis: str,
    board_summary: str,
    segment: pd.DataFrame,
    strategy_df: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot": snapshot,
        "premium_basis": premium_basis,
        "pressure_label": pressure_label,
        "phase6_metrics": phase6_metrics,
        "ep_metrics": ep_metrics,
        "governance_checks": governance,
        "governance_ok": governance_ok,
        "capital_metrics": {
            "capital_required": capital_required,
            "capital_available": capital_available,
            "solvency_ratio": solvency_ratio,
        },
        "risk_analysis": risk_analysis,
        "capital_analysis": capital_analysis,
        "board_summary": board_summary,
        "segment": segment.to_dict(orient="records"),
        "top_actions": strategy_df.head(5).to_dict(orient="records"),
    }


def save_payload(payload: dict[str, Any], out_path: Path) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def answer_portfolio_question(question: str, payload: dict[str, Any], docs: list[dict[str, str]]) -> str:
    q = normalize_text(question)

    cap = payload["capital_metrics"]
    ep = payload["ep_metrics"]
    gov = payload["governance_checks"]

    if "capital" in q:
        return (
            f"No. The portfolio is not adequately capitalised. "
            f"Capital required is {int(cap['capital_required']):,}, capital available is {int(cap['capital_available']):,}, "
            f"and solvency_ratio is {cap['solvency_ratio']:.4f}. "
            f"The 1-in-200 loss is {ep['loss_1_in_200_bn']:.4f} bn."
        )

    if "tail" in q or "1_in_200" in q or "1 in 200" in q:
        return (
            f"The portfolio 1-in-200 loss is {ep['loss_1_in_200_bn']:.4f} bn and TVaR 99 is {ep['tvar_99_bn']:.4f} bn, "
            f"indicating significant tail-risk exposure."
        )

    if "governance" in q or "control" in q:
        return (
            f"Governance status is {'passed' if payload['governance_ok'] else 'not fully passed'}. "
            f"Checks: exposure_valid={gov.get('exposure_valid')}, "
            f"frequency_valid={gov.get('frequency_valid')}, "
            f"fraud_layer_present={gov.get('fraud_layer_present')}, "
            f"scenario_defined={gov.get('scenario_defined')}."
        )

    if "price" in q or "underwrite" in q or "risk" in q:
        top = payload["top_actions"][0]
        return (
            f"The highest-priority action is {top['action']} in {top['product_type']} / {top['channel']}. "
            f"Recommended_rate_change_pct is {top['recommended_rate_change_pct']:.4f}. "
            f"Rationale: {top['rationale']}"
        )

    matches = retrieve_context(question, docs, top_k=3)
    return "\n\n".join([f"[{m['source']}]\n{m['text']}" for m in matches])