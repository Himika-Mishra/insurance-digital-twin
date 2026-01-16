"""
Core synthetic data generators for the insurance digital twin.

Design goals:
- Statistically coherent with real-world personal-lines portfolios.
- Product-level frequencies and severities approx match TARGET_* in config.
- Risk factors (age, region, risk_profile, coverage, channel) actually matter.
- Macro shocks (e.g. 2020, 2022) affect severity and frequency.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Tuple

import numpy as np
import pandas as pd

from . import config


# ---------------- Utility helpers ---------------- #

def _random_dates(n: int, start_year: int, end_year: int, rng: np.random.Generator) -> list[datetime]:
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta_days = (end - start).days
    return [
        start + timedelta(days=int(d))
        for d in rng.integers(0, delta_days, size=n)
    ]


# ---------------- Macro layer ---------------- #

def generate_macro() -> pd.DataFrame:
    """Monthly macro indicators with base drift + shocks.

    Returns:
        DataFrame with columns:
        [month, inflation_index, unemployment_rate, repair_cost_index, catastrophe_flag]
    """
    rng = np.random.default_rng(config.SEED_MACRO)

    dates = pd.date_range(
        f"{config.START_YEAR}-01-01",
        f"{config.END_YEAR}-12-01",
        freq="MS",
    )

    infl, unemp, repair_idx, cat_flag = [], [], [], []
    level = config.MACRO_BASE_INFLATION

    for d in dates:
        shock = 0.0
        if d.year in config.MACRO_SHOCK_YEARS:
            # inflation shock, e.g. covid / supply chain issues
            shock = rng.normal(0.05, 0.02)

        # AR(1)-like drift
        level = level * (1 + config.MACRO_ANNUAL_INFLATION_DRIFT / 12) + shock
        level = max(level, 0.5)
        infl.append(level)

        unemp_val = np.clip(rng.normal(0.06, 0.01), 0.03, 0.12)
        unemp.append(unemp_val)

        repair_idx.append(level * rng.uniform(0.9, 1.15))
        cat_flag.append(int(rng.random() < 0.02))  # ~2% of months with cat-like event

    macro = pd.DataFrame(
        {
            "month": dates,
            "inflation_index": infl,
            "unemployment_rate": unemp,
            "repair_cost_index": repair_idx,
            "catastrophe_flag": cat_flag,
        }
    )
    return macro


# ---------------- Policyholders ---------------- #

def generate_policyholders(n: int) -> pd.DataFrame:
    """Generate a realistic pool of policyholders with risk profiles."""
    rng = np.random.default_rng(config.SEED_POLICYHOLDERS)

    ages = rng.integers(18, 85, size=n)
    genders = rng.choice(["M", "F"], size=n, p=[0.52, 0.48])
    regions = rng.choice(
        config.REGIONS,
        size=n,
        p=[0.22, 0.25, 0.18, 0.20, 0.15],  # slightly skewed by population
    )
    income_band = rng.choice(
        ["low", "lower_mid", "upper_mid", "high"],
        size=n,
        p=[0.25, 0.35, 0.25, 0.15],
    )
    occupations = rng.choice(
        ["clerical", "manual", "professional", "self_employed", "student", "retired"],
        size=n,
    )

    # Continuous underlying risk score (not shown to models directly)
    base_risk = (
        (ages < 25).astype(float) * 0.8
        + (ages > 70).astype(float) * 0.6
        + (income_band == "low").astype(float) * 0.6
        + np.isin(regions, ["north", "midlands"]).astype(float) * 0.4
    )
    base_risk += rng.normal(0.0, 0.3, size=n)

    # Map into risk_profile buckets using quantiles
    q_low, q_med = np.quantile(base_risk, [0.4, 0.8])
    risk_profile = np.where(
        base_risk <= q_low,
        "low",
        np.where(base_risk <= q_med, "medium", "high"),
    )

    holders = pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1),
            "age": ages,
            "gender": genders,
            "region": regions,
            "income_band": income_band,
            "occupation": occupations,
            "risk_profile": risk_profile,
            "raw_risk_score": base_risk,
        }
    )
    return holders


# ---------------- Policies ---------------- #

def generate_policies(policyholders: pd.DataFrame, n_policies: int) -> pd.DataFrame:
    """Generate realistic policies linked to policyholders."""
    rng = np.random.default_rng(config.SEED_POLICIES)

    customer_ids = rng.choice(policyholders["customer_id"], size=n_policies, replace=True)

    products = rng.choice(
        config.PRODUCT_TYPES,
        size=n_policies,
        p=[config.PRODUCT_MIX[p] for p in config.PRODUCT_TYPES],
    )
    coverage_level = rng.choice(
        config.COVERAGE_LEVELS,
        size=n_policies,
        p=[0.4, 0.4, 0.2],
    )
    channel = rng.choice(config.CHANNELS, size=n_policies, p=[0.55, 0.30, 0.15])

    start_dates = _random_dates(n_policies, config.START_YEAR, config.END_YEAR - 1, rng)
    end_dates = [d + timedelta(days=365) for d in start_dates]

    sum_insured = []
    premiums = []
    vehicle_age = []

    for prod, cov in zip(products, coverage_level):
        # baseline sums insured by product
        if prod == "motor":
            base_si = rng.uniform(6_000, 25_000)
            veh_age = int(rng.integers(0, 15))
        elif prod == "gap":
            base_si = rng.uniform(3_000, 15_000)
            veh_age = int(rng.integers(0, 8))
        elif prod == "home":
            base_si = rng.uniform(80_000, 400_000)
            veh_age = 0
        elif prod == "warranty":
            base_si = rng.uniform(500, 3_000)
            veh_age = int(rng.integers(0, 10))
        else:  # health
            base_si = rng.uniform(20_000, 150_000)
            veh_age = 0

        cov_mult = {"basic": 0.8, "standard": 1.0, "premium": 1.3}[cov]
        si = base_si * cov_mult
        sum_insured.append(si)
        vehicle_age.append(veh_age)

        target_lr = config.TARGET_LOSS_RATIO[prod]
        freq = config.TARGET_FREQ[prod]
        sev = config.TARGET_SEVERITY[prod]

        # technical premium ~ freq * sev / LR
        tech_prem = freq * sev / max(target_lr, 0.01)
        # scale by coverage & SI with diminishing returns
        tech_prem *= cov_mult * (si / (base_si + 1e-6)) ** 0.3
        premium = tech_prem * rng.uniform(0.85, 1.15)

        premiums.append(premium)

    deductibles = np.array(sum_insured) * np.random.default_rng(config.SEED_POLICIES + 1).uniform(
        0.01, 0.05, size=n_policies
    )

    policies = pd.DataFrame(
        {
            "policy_id": np.arange(1, n_policies + 1),
            "customer_id": customer_ids,
            "product_type": products,
            "coverage_level": coverage_level,
            "sum_insured": sum_insured,
            "deductible": deductibles,
            "start_date": start_dates,
            "end_date": end_dates,
            "channel": channel,
            "base_annual_premium": premiums,
            "vehicle_age": vehicle_age,
        }
    )
    return policies


# ---------------- Monthly exposure ---------------- #

# def build_exposure_table(policies: pd.DataFrame) -> pd.DataFrame:
#     """Expand policies into monthly exposure rows within the global macro horizon."""
#     rng = np.random.default_rng(config.SEED_EXPOSURE)

#     rows = []
#     macro_start = pd.Period(f"{config.START_YEAR}-01", freq="M")
#     macro_end = pd.Period(f"{config.END_YEAR}-12", freq="M")

#     for _, pol in policies.iterrows():
#         start_month = pd.Period(pol["start_date"].strftime("%Y-%m"), freq="M")
#         end_month = pd.Period(pol["end_date"].strftime("%Y-%m"), freq="M")

#         # clamp to macro horizon
#         start_month = max(start_month, macro_start)
#         end_month = min(end_month, macro_end)
#         if end_month < start_month:
#             continue

#         months = pd.period_range(start_month, end_month, freq="M")
#         for m in months:
#             rows.append(
#                 {
#                     "policy_id": pol["policy_id"],
#                     "customer_id": pol["customer_id"],
#                     "product_type": pol["product_type"],
#                     "coverage_level": pol["coverage_level"],
#                     "channel": pol["channel"],
#                     "month": m.to_timestamp(),
#                     "exposure": 1.0,  # full month for simplicity
#                 }
#             )

#     expo = pd.DataFrame(rows)
#     return expo

def build_exposure_table(policies: pd.DataFrame) -> pd.DataFrame:
    """
    Expand policies into monthly exposure rows with:
    - partial months
    - mid-term cancellations
    - lapse probability
    """
    rng = np.random.default_rng(config.SEED_EXPOSURE)

    rows = []
    macro_start = pd.Period(f"{config.START_YEAR}-01", freq="M")
    macro_end = pd.Period(f"{config.END_YEAR}-12", freq="M")

    for _, pol in policies.iterrows():
        start = pd.Timestamp(pol["start_date"])
        end = pd.Timestamp(pol["end_date"])

        # --- Cancellation logic ---
        base_cancel_prob = 0.08
        if pol["channel"] == "aggregator":
            base_cancel_prob += 0.06
        if pol["coverage_level"] == "basic":
            base_cancel_prob += 0.03

        cancels = rng.random() < base_cancel_prob
        if cancels:
            cancel_months = int(rng.integers(2, 10))
            end = min(end, start + pd.DateOffset(months=cancel_months))

        start_m = pd.Period(start.strftime("%Y-%m"), freq="M")
        end_m = pd.Period(end.strftime("%Y-%m"), freq="M")

        start_m = max(start_m, macro_start)
        end_m = min(end_m, macro_end)
        if end_m < start_m:
            continue

        months = pd.period_range(start_m, end_m, freq="M")

        for i, m in enumerate(months):
            exposure = 1.0

            # partial first month
            if i == 0:
                exposure *= rng.uniform(0.3, 1.0)

            # partial last month (if cancelled)
            if i == len(months) - 1 and cancels:
                exposure *= rng.uniform(0.2, 0.8)

            # random zero exposure (payment failure / admin)
            if rng.random() < 0.01:
                exposure = 0.0

            rows.append(
                {
                    "policy_id": pol["policy_id"],
                    "customer_id": pol["customer_id"],
                    "product_type": pol["product_type"],
                    "coverage_level": pol["coverage_level"],
                    "channel": pol["channel"],
                    "month": m.to_timestamp(),
                    "exposure": round(exposure, 3),
                }
            )

    return pd.DataFrame(rows)



# ---------------- Frequency & severity models ---------------- #

# modified for the overdisposition
def _frequency_lambda(
    expo_row: pd.Series,
    policy: pd.Series,
    holder: pd.Series,
    macro_row: pd.Series,
    frailty: float,
    rng: np.random.Generator,
) -> float:
    """
    GLM-style log-link for monthly claim frequency with latent frailty:

    log(lambda) = intercept(prod)
                  + beta_risk_profile
                  + beta_age
                  + beta_channel
                  + beta_macro
                  + log(frailty)
                  + noise

    Where 'frailty' is a policy-level Gamma heterogeneity term, which
    induces overdispersion (Poisson-Gamma mixture ≈ Negative Binomial).
    """
    prod = expo_row["product_type"]
    risk = holder["risk_profile"]
    age = holder["age"]
    channel = expo_row["channel"]

    # base monthly intensity from annual target
    base_annual = config.TARGET_FREQ[prod]
    base_monthly = base_annual / 12.0
    base_monthly = max(base_monthly, 1e-4)  # guard

    log_lambda = np.log(base_monthly)

    # risk profile
    if risk == "medium":
        log_lambda += 0.25
    elif risk == "high":
        log_lambda += 0.55

    # age effects (motor more sensitive)
    if prod == "motor" and age < 25:
        log_lambda += 0.6
    if prod == "motor" and age > 70:
        log_lambda += 0.3

    # channel: aggregators → slightly higher freq, direct slightly lower
    if channel == "aggregator":
        log_lambda += 0.15
    elif channel == "direct":
        log_lambda -= 0.05

    # macro: catastrophe months more claims
    if macro_row["catastrophe_flag"] == 1:
        log_lambda += 0.4

    # strong winter vs summer effect
    month_num = expo_row["month"].month
    if month_num in (11, 12, 1, 2):
        log_lambda += 0.5      # winter bump
    elif month_num in (6, 7, 8):
        log_lambda -= 0.3      # quieter summer

    # policy-level frailty (Gamma heterogeneity) – multiplicative on lambda
    frailty = max(float(frailty), 1e-3)
    log_lambda += np.log(frailty)

    # extra random heterogeneity
    log_lambda += rng.normal(0.0, 1.7)

    lam = float(np.exp(log_lambda))
    return lam


def _severity_mean(
    prod: str,
    coverage: str,
    sum_insured: float,
    macro_row: pd.Series,
    risk: str,
) -> float:
    """Expected claim severity given product, coverage, macro and risk."""
    base = config.TARGET_SEVERITY[prod]
    cov_mult = {"basic": 0.9, "standard": 1.0, "premium": 1.25}[coverage]

    macro_mult = (macro_row["inflation_index"] + macro_row["repair_cost_index"]) / 2.0
    macro_mult = max(macro_mult, 0.7)

    risk_mult = 1.0
    if risk == "medium":
        risk_mult = 1.1
    elif risk == "high":
        risk_mult = 1.25

    # scale with SI with diminishing returns
    si_mult = (sum_insured / (base * 5.0)) ** 0.3
    si_mult = float(np.clip(si_mult, 0.7, 1.8))

    return float(base * cov_mult * macro_mult * risk_mult * si_mult)


# ---------------- Claims ---------------- #

def generate_claims(
    policies: pd.DataFrame,
    policyholders: pd.DataFrame,
    macro: pd.DataFrame,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Generate claims using realistic frequency & severity models."""
    seed = random_state if random_state is not None else config.SEED_CLAIMS
    rng = np.random.default_rng(seed)

    macro = macro.copy()
    macro["ym"] = macro["month"].dt.to_period("M")
    macro_map = macro.set_index("ym")

    expo = build_exposure_table(policies)
    expo["ym"] = expo["month"].dt.to_period("M")

    holders = policyholders.set_index("customer_id")
    policies_indexed = policies.set_index("policy_id")

    # Policy-level latent frailty (Gamma) to induce overdispersion
    policy_frailty = pd.Series(
        rng.gamma(shape=0.7, scale=1.8, size=len(policies_indexed)),
        index=policies_indexed.index,
        name="frailty",
    )

    multi_claim_factor = pd.Series(
        rng.choice(
            [1.0, 1.5, 2.0, 3.0],
            p=[0.85, 0.08, 0.05, 0.02],
            size=len(policies_indexed),
        ),
        index=policies_indexed.index,
        name="multi_factor",
    )

    claim_rows: list[dict] = []

    for _, row in expo.iterrows():
        pol = policies_indexed.loc[row["policy_id"]]
        holder = holders.loc[row["customer_id"]]
        ym = row["ym"]

        if ym not in macro_map.index:
            continue
        macro_row = macro_map.loc[ym]

        frailty = policy_frailty.loc[row["policy_id"]] * multi_claim_factor.loc[row["policy_id"]]
        lam = _frequency_lambda(row, pol, holder, macro_row, frailty, rng)
        n_claims = rng.poisson(lam * row["exposure"])

        if n_claims == 0:
            continue

        for _ in range(n_claims):
            # random day in that exposure month
            month_start = ym.to_timestamp()
            incident_date = month_start + timedelta(days=int(rng.integers(0, 28)))
            reported_delay = int(rng.integers(0, 40))
            reported_date = incident_date + timedelta(days=reported_delay)

            mean_sev = _severity_mean(
                prod=pol["product_type"],
                coverage=pol["coverage_level"],
                sum_insured=pol["sum_insured"],
                macro_row=macro_row,
                risk=holder["risk_profile"],
            )

            sigma = 0.6  # heavy right tail
            mu = np.log(mean_sev) - 0.5 * sigma ** 2
            severity = rng.lognormal(mean=mu, sigma=sigma)

            # fraud probability depending on size & delay
            is_fraud = int(
                (severity > mean_sev * 3 and rng.random() < 0.35)
                or (reported_delay > 25 and rng.random() < 0.2)
            )

            fraud_ring_id = (
                int(rng.integers(1, 300))
                if is_fraud and rng.random() < 0.5
                else None
            )

            claim_type = rng.choice(
                ["accident", "theft", "fire", "mechanical", "medical"],
                p=[0.5, 0.15, 0.10, 0.15, 0.10],
            )

            status = rng.choice(
                ["open", "closed", "repudiated"],
                p=[0.18, 0.72, 0.10],
            )

            paid = severity * rng.uniform(0.6, 1.0) if status != "repudiated" else 0.0
            outstanding = 0.0 if status != "open" else max(severity - paid, 0.0)

            claim_rows.append(
                {
                    "policy_id": int(row["policy_id"]),
                    "customer_id": int(row["customer_id"]),
                    "incident_date": incident_date,
                    "reported_date": reported_date,
                    "claim_type": claim_type,
                    "cause": "synthetic_engine",
                    "status": status,
                    "paid_amount": float(paid),
                    "outstanding_reserve": float(outstanding),
                    "is_fraud": is_fraud,
                    "fraud_ring_id": fraud_ring_id,
                }
            )

    claims_df = pd.DataFrame(claim_rows)
    if not claims_df.empty:
        claims_df.insert(0, "claim_id", np.arange(1, len(claims_df) + 1))
    else:
        claims_df = pd.DataFrame(
            columns=[
                "claim_id",
                "policy_id",
                "customer_id",
                "incident_date",
                "reported_date",
                "claim_type",
                "cause",
                "status",
                "paid_amount",
                "outstanding_reserve",
                "is_fraud",
                "fraud_ring_id",
            ]
        )
    return claims_df


# ---------------- Universe entrypoint ---------------- #

def inject_anomalies(
    macro: pd.DataFrame,
    holders: pd.DataFrame,
    policies: pd.DataFrame,
    claims: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Inject realistic data anomalies into the synthetic universe.

    These are intentional imperfections to mimic real-world data issues:
    - Missing values
    - Bad dates
    - Pricing & exposure quirks
    - Severity outliers
    - Inconsistent fraud / status patterns
    """
    rng = np.random.default_rng(config.SEED_ANOMALIES)

    macro = macro.copy()
    holders = holders.copy()
    policies = policies.copy()
    claims = claims.copy()

    # ---------------- Policyholder anomalies ---------------- #

    # ~1% missing region, ~1% missing income_band
    if len(holders) > 0:
        n_missing_region = max(1, int(0.01 * len(holders)))
        idx_region = rng.choice(holders.index, size=n_missing_region, replace=False)
        holders.loc[idx_region, "region"] = None

        n_missing_income = max(1, int(0.01 * len(holders)))
        idx_income = rng.choice(holders.index, size=n_missing_income, replace=False)
        holders.loc[idx_income, "income_band"] = None

        # A few extreme ages (e.g. 110, 5 years old – data coding issues)
        n_weird_age = max(1, int(0.001 * len(holders)))
        idx_age = rng.choice(holders.index, size=n_weird_age, replace=False)
        holders.loc[idx_age, "age"] = rng.choice([5, 110], size=n_weird_age)

    # ---------------- Policy anomalies ---------------- #

    if len(policies) > 0:
        # Some zero or tiny premiums (underpriced / data entry)
        n_zero_prem = max(1, int(0.002 * len(policies)))
        idx_zero = rng.choice(policies.index, size=n_zero_prem, replace=False)
        policies.loc[idx_zero, "base_annual_premium"] = rng.choice([0.0, 1.0], size=n_zero_prem)

        # Zero deductibles for some premium policies (unusually generous or miskey)
        mask_prem = policies["coverage_level"] == "premium"
        idx_zero_ded = policies[mask_prem].sample(
            n=min( max(1, int(0.01 * mask_prem.sum())), mask_prem.sum() ),
            random_state=config.SEED_ANOMALIES,
        ).index
        policies.loc[idx_zero_ded, "deductible"] = 0.0

        # A few crazy high sum insured values (e.g. one-off mansion / supercar)
        n_high_si = max(1, int(0.002 * len(policies)))
        idx_high_si = rng.choice(policies.index, size=n_high_si, replace=False)
        policies.loc[idx_high_si, "sum_insured"] *= rng.uniform(10, 25, size=n_high_si)

        # A tiny set of policies with end_date before start_date
        n_bad_dates = max(1, int(0.001 * len(policies)))
        idx_bad = rng.choice(policies.index, size=n_bad_dates, replace=False)
        policies.loc[idx_bad, "end_date"] = policies.loc[idx_bad, "start_date"] - pd.to_timedelta(
            rng.integers(1, 60, size=n_bad_dates), unit="D"
        )

    # ---------------- Macro anomalies ---------------- #

    if len(macro) > 0:
        # One "crazy" inflation month (measurement or coding glitch)
        idx_glitch = rng.integers(0, len(macro))
        macro.loc[macro.index[idx_glitch], "inflation_index"] *= rng.uniform(2.5, 4.0)

    # ---------------- Claims anomalies ---------------- #

    if len(claims) > 0:
        # Negative paid_amount (reversals / data errors)
        n_negative = max(1, int(0.005 * len(claims)))
        idx_neg = rng.choice(claims.index, size=n_negative, replace=False)
        claims.loc[idx_neg, "paid_amount"] = -np.abs(
            claims.loc[idx_neg, "paid_amount"].values * rng.uniform(0.2, 1.0, size=n_negative)
        )

        # Reported before incident date (date swap)
        n_bad_dates = max(1, int(0.003 * len(claims)))
        idx_bad = rng.choice(claims.index, size=n_bad_dates, replace=False)
        # subtract 1–30 days from reported_date relative to incident_date
        claims.loc[idx_bad, "reported_date"] = claims.loc[idx_bad, "incident_date"] - pd.to_timedelta(
            rng.integers(1, 30, size=n_bad_dates), unit="D"
        )

        # A few extreme severity outliers
        n_outliers = max(1, int(0.003 * len(claims)))
        idx_out = rng.choice(claims.index, size=n_outliers, replace=False)
        claims.loc[idx_out, "paid_amount"] *= rng.uniform(8, 15, size=n_outliers)

        # Some 'repudiated' claims with positive paid_amount (inconsistent coding)
        mask_rep = claims["status"] == "repudiated"
        if mask_rep.sum() > 0:
            n_rep_paid = max(1, int(0.05 * mask_rep.sum()))
            idx_rep_paid = claims[mask_rep].sample(
                n=min(n_rep_paid, mask_rep.sum()),
                random_state=config.SEED_ANOMALIES + 1,
            ).index
            claims.loc[idx_rep_paid, "paid_amount"] = np.abs(
                claims.loc[idx_rep_paid, "paid_amount"].values
            ) + rng.uniform(50, 500, size=len(idx_rep_paid))

        # Recalculate outstanding_reserve a bit inconsistently for some rows
        n_reserve_weird = max(1, int(0.01 * len(claims)))
        idx_reserve = rng.choice(claims.index, size=n_reserve_weird, replace=False)
        claims.loc[idx_reserve, "outstanding_reserve"] *= rng.uniform(-0.5, 2.0, size=n_reserve_weird)

    return macro, holders, policies, claims

def generate_universe(
    target_overall_lr: float = 0.70,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate the full "insurance universe", calibrate premiums
    to a target overall loss ratio, and then inject realistic anomalies.

    Returns:
        macro_df, policyholders_df, policies_df, claims_df
    """
    # 1) Generate clean synthetic data
    macro = generate_macro()
    holders = generate_policyholders(config.N_POLICYHOLDERS)
    policies = generate_policies(holders, config.N_POLICIES)
    claims = generate_claims(policies, holders, macro)

    if claims.empty:
        # Nothing to calibrate or corrupt
        return macro, holders, policies, claims

    # 2) Calibrate premiums to target LR on clean data
    total_premium = policies["base_annual_premium"].sum()
    total_paid = claims["paid_amount"].sum()
    actual_lr = total_paid / max(total_premium, 1e-6)

    if actual_lr > 0:
        factor = actual_lr / target_overall_lr
        policies["base_annual_premium"] *= factor

    # 3) Inject anomalies to mimic real-world data issues
    macro, holders, policies, claims = inject_anomalies(macro, holders, policies, claims)

    return macro, holders, policies, claims