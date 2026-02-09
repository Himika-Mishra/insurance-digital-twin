
import numpy as np
import pandas as pd
import streamlit as st
from dataclasses import dataclass

# --- The notebook must export these objects or you must re-load them here ---
# We assume you will run this app from the project root and will load artifacts.
# Option A (recommended): re-load frozen data + rebuild the engine inside this app.
# Option B: import engine from a module. We'll do A for reliability.

import statsmodels.api as sm

DATA_PATH = "data/raw"

# Load frozen data
macro = pd.read_csv(f"{DATA_PATH}/macro.csv", parse_dates=["month"])
policies = pd.read_csv(f"{DATA_PATH}/policies.csv", parse_dates=["start_date","end_date"])
claims = pd.read_csv(f"{DATA_PATH}/claims.csv", parse_dates=["incident_date","reported_date"])

# Build macro join
macro["ym"] = pd.to_datetime(macro["month"]).dt.to_period("M")
claims["ym"] = pd.to_datetime(claims["incident_date"]).dt.to_period("M")

clm = claims.merge(macro[["ym","inflation_index","repair_cost_index","unemployment_rate","catastrophe_flag"]],
                   on="ym", how="left", validate="many_to_one") \
           .merge(policies[["policy_id","product_type"]], on="policy_id", how="left", validate="many_to_one")

# Severity dataset
sev = clm.loc[clm["paid_amount"] > 0].copy()
p_lo, p_hi = sev["paid_amount"].quantile([0.01, 0.99])
sev["paid_w"] = sev["paid_amount"].clip(lower=p_lo, upper=p_hi)

sev["log_paid"] = np.log(sev["paid_w"])
sev["log_infl"] = np.log(sev["inflation_index"])
sev["log_repair"] = np.log(sev["repair_cost_index"])

X = sm.add_constant(sev[["log_infl","log_repair","unemployment_rate","catastrophe_flag"]])
y = sev["log_paid"]
sev_model = sm.OLS(y, X).fit(cov_type="HC3")

# Frequency dataset (month x product)
freq = clm.groupby(["product_type","ym"]).agg(
    claim_count=("claim_id","count"),
    inflation=("inflation_index","mean"),
    repair=("repair_cost_index","mean"),
    unemp=("unemployment_rate","mean"),
    cat=("catastrophe_flag","max")
).reset_index()
freq["log_infl"] = np.log(freq["inflation"])
freq["log_repair"] = np.log(freq["repair"])

df = freq.dropna(subset=["claim_count","log_infl","log_repair","unemp","cat"]).copy()

prod_dum = pd.get_dummies(df["product_type"], drop_first=True, prefix="prod", dtype=float)
Xf = pd.concat([df[["cat","log_infl","log_repair","unemp"]].astype(float), prod_dum], axis=1)
Xf = sm.add_constant(Xf, has_constant="add")
yf = df["claim_count"].astype(int)

pois = sm.GLM(yf, Xf, family=sm.families.Poisson()).fit()
mu = pois.fittedvalues
pearson_chi2 = np.sum(((yf - mu) ** 2) / np.maximum(mu, 1e-9))
dispersion = pearson_chi2 / pois.df_resid
alpha = max(dispersion - 1.0, 0.1)

nb = sm.GLM(yf, Xf, family=sm.families.NegativeBinomial(alpha=alpha)).fit(cov_type="HC3")

# Monthly baseline for scenario engine
monthly_prod = clm.assign(paid_clean=lambda d: d["paid_amount"].where(d["paid_amount"] > 0, np.nan)) \
    .groupby(["product_type","ym"]).agg(
        claim_count=("claim_id","count"),
        mean_severity=("paid_clean","mean"),
        inflation=("inflation_index","mean"),
        repair=("repair_cost_index","mean"),
        unemp=("unemployment_rate","mean"),
        cat=("catastrophe_flag","max")
    ).reset_index()
monthly_prod["base_total_paid"] = monthly_prod["claim_count"] * monthly_prod["mean_severity"]

@dataclass(frozen=True)
class Scenario:
    name: str
    infl_mult: float = 1.0
    repair_mult: float = 1.0
    unemp_shift: float = 0.0
    cat_override: int | None = None

@dataclass(frozen=True)
class ReinsuranceProgram:
    qs_share: float = 0.0
    xl_retention: float | None = None
    xl_limit: float | None = None

def apply_reinsurance(gross_paid: float, program: ReinsuranceProgram) -> dict:
    gross_paid = float(gross_paid)
    ceded_qs = gross_paid * float(program.qs_share)
    retained = gross_paid - ceded_qs
    ceded_xl = 0.0
    if program.xl_retention is not None and program.xl_limit is not None:
        xs = max(retained - float(program.xl_retention), 0.0)
        ceded_xl = min(xs, float(program.xl_limit))
    net = retained - ceded_xl
    return {"gross": gross_paid, "net": net, "ceded_qs": ceded_qs, "ceded_xl": ceded_xl}

class ScenarioEngine:
    def __init__(self, monthly_prod_df, sev_model, nb_model):
        self.base = monthly_prod_df.copy()
        self.sev_model = sev_model
        self.nb_model = nb_model
        self.b_freq = nb_model.params.to_dict()
        self.b_sev = sev_model.params.to_dict()

    def _freq_mult(self, row, scen: Scenario):
        d = 0.0
        d += float(self.b_freq.get("log_infl", 0.0)) * np.log(scen.infl_mult)
        d += float(self.b_freq.get("log_repair", 0.0)) * np.log(scen.repair_mult)
        d += float(self.b_freq.get("unemp", 0.0)) * float(scen.unemp_shift)
        if scen.cat_override is not None:
            d += float(self.b_freq.get("cat", 0.0)) * (scen.cat_override - row["cat"])
        return float(np.exp(d))

    def _sev_mult(self, scen: Scenario):
        d = 0.0
        d += float(self.b_sev.get("log_infl", 0.0)) * np.log(scen.infl_mult)
        d += float(self.b_sev.get("log_repair", 0.0)) * np.log(scen.repair_mult)
        d += float(self.b_sev.get("unemployment_rate", 0.0)) * float(scen.unemp_shift)
        return float(np.exp(d))

    def run(self, scen: Scenario):
        df = self.base.copy()
        sev_mult = self._sev_mult(scen)
        df["freq_mult"] = df.apply(lambda r: self._freq_mult(r, scen), axis=1)
        df["sev_mult"] = sev_mult
        df["proj_paid"] = (df["claim_count"] * df["freq_mult"]) * (df["mean_severity"] * df["sev_mult"])

        by_prod = df.groupby("product_type").agg(base_paid=("base_total_paid","sum"), scen_paid=("proj_paid","sum"))
        by_prod["pct_change"] = (by_prod["scen_paid"] / by_prod["base_paid"]) - 1.0

        base_total = float(by_prod["base_paid"].sum())
        scen_total = float(by_prod["scen_paid"].sum())

        return by_prod.sort_values("pct_change", ascending=False), base_total, scen_total

engine = ScenarioEngine(monthly_prod, sev_model, nb)

# ---------------- UI ----------------
st.set_page_config(page_title="Insurance Digital Twin — Scenario Simulator", layout="wide")
st.title("Insurance Digital Twin — Macro & CAT Scenario Simulator (Exec Demo)")

col1, col2, col3 = st.columns(3)
with col1:
    infl = st.slider("Inflation shock (%)", 0, 30, 10)
    repair = st.slider("Repair cost shock (%)", 0, 30, 10)
with col2:
    unemp_pp = st.slider("Unemployment shift (pp)", -3.0, 3.0, 1.0, 0.1)
    cat = st.checkbox("Force CAT year", value=False)
with col3:
    st.subheader("Reinsurance (optional)")
    qs = st.slider("Quota share (%)", 0, 50, 25) / 100.0
    xl_ret = st.number_input("XL retention (£)", value=5_000_000, step=500_000)
    xl_lim = st.number_input("XL limit (£)", value=15_000_000, step=500_000)

scen = Scenario(
    name="Live scenario",
    infl_mult=1.0 + infl/100.0,
    repair_mult=1.0 + repair/100.0,
    unemp_shift=unemp_pp/100.0,   # convert pp to rate shift
    cat_override=1 if cat else None
)

by_prod, base_total, scen_total = engine.run(scen)
gross = scen_total
ri = apply_reinsurance(gross, ReinsuranceProgram(qs_share=qs, xl_retention=xl_ret, xl_limit=xl_lim))

st.metric("Baseline Paid (gross)", f"£{base_total:,.0f}")
st.metric("Scenario Paid (gross)", f"£{gross:,.0f}")
st.metric("Scenario Paid (net after RI)", f"£{ri['net']:,.0f}")

st.caption(f"Ceded QS: £{ri['ceded_qs']:,.0f} | Ceded XL: £{ri['ceded_xl']:,.0f}")

st.subheader("Product attribution (gross)")
out = by_prod.copy()
out["base_paid"] = out["base_paid"].map(lambda x: f"£{x:,.0f}")
out["scen_paid"] = out["scen_paid"].map(lambda x: f"£{x:,.0f}")
out["pct_change"] = (by_prod["pct_change"]*100).map(lambda x: f"{x:.1f}%")
st.dataframe(out, use_container_width=True)
