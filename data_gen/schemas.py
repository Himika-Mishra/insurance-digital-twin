"""
Schema definitions for core entities in the synthetic insurance universe.

These schemas define the **contract** between:
- data generation
- validation / quality gates
- downstream pricing & risk models

They are intentionally kept lightweight and mirror the pandas DataFrame
structures used throughout the project.
"""

from dataclasses import dataclass
from typing import Optional
from datetime import date


# ---------------- Policyholder ---------------- #

@dataclass
class PolicyholderSchema:
    customer_id: int
    age: int
    gender: str
    region: str
    income_band: str
    occupation: str
    risk_profile: str   # categorical: "low" | "medium" | "high"
    raw_risk_score: float


# ---------------- Policy ---------------- #

@dataclass
class PolicySchema:
    policy_id: int
    customer_id: int
    product_type: str   # motor / gap / home / warranty / health
    coverage_level: str # basic / standard / premium
    sum_insured: float
    deductible: float
    start_date: date
    end_date: date
    channel: str        # broker / direct / aggregator
    base_annual_premium: float
    vehicle_age: int    # 0 for non-motor/warranty


# ---------------- Monthly Exposure ---------------- #

@dataclass
class ExposureSchema:
    policy_id: int
    customer_id: int
    product_type: str
    coverage_level: str
    channel: str
    month: date
    exposure: float     # 0–1 fraction of month


# ---------------- Claim ---------------- #

@dataclass
class ClaimSchema:
    claim_id: int
    policy_id: int
    customer_id: int
    incident_date: date
    reported_date: date
    claim_type: str     # accident/theft/fire/mechanical/medical
    cause: str          # synthetic cause tag
    status: str         # open / closed / repudiated
    paid_amount: float
    outstanding_reserve: float
    is_fraud: int
    fraud_ring_id: Optional[int]


# ---------------- Macro ---------------- #

@dataclass
class MacroSchema:
    month: date
    inflation_index: float
    unemployment_rate: float
    repair_cost_index: float
    catastrophe_flag: int
