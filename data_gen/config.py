"""
Configuration for synthetic insurance universe generation.

These parameters are chosen to be broadly realistic for a
personal-lines portfolio in a developed market (e.g. UK/EU).
"""

from datetime import date

# ---------------- Portfolio size ---------------- #

# You can scale these up/down depending on your hardware
N_POLICYHOLDERS: int = 80_000
N_POLICIES: int = 120_000

# Policy/claim time horizon
START_YEAR: int = 2018
END_YEAR: int = 2024  # inclusive


# ---------------- Dimensions & categories ---------------- #

PRODUCT_TYPES = ["motor", "gap", "home", "warranty", "health"]
REGIONS = ["north", "south", "east", "west", "midlands"]
CHANNELS = ["broker", "direct", "aggregator"]
COVERAGE_LEVELS = ["basic", "standard", "premium"]

# Approximate portfolio mix by product
PRODUCT_MIX = {
    "motor": 0.45,
    "gap": 0.08,
    "home": 0.20,
    "warranty": 0.15,
    "health": 0.12,
}

# ---------------- Target technical metrics ---------------- #

# Annual claim frequency per policy (per product)
TARGET_FREQ = {
    "motor": 0.12,
    "gap": 0.05,
    "home": 0.08,
    "warranty": 0.16,
    "health": 0.28,
}

# Mean severity per claim, approximate GBP values
TARGET_SEVERITY = {
    "motor": 1_800,
    "gap": 2_400,
    "home": 6_000,
    "warranty": 750,
    "health": 2_200,
}

# Target loss ratio (claims / premium)
TARGET_LOSS_RATIO = {
    "motor": 0.68,
    "gap": 0.55,
    "home": 0.60,
    "warranty": 0.70,
    "health": 0.72,
}

# ---------------- Macro environment ---------------- #

MACRO_BASE_INFLATION: float = 1.0
MACRO_ANNUAL_INFLATION_DRIFT: float = 0.025  # ~2.5% per year baseline

# Years with shocks (e.g. covid, supply-chain, inflation spikes)
MACRO_SHOCK_YEARS = [2020, 2022]

# ---------------- Fraud controls ---------------- #

BASE_FRAUD_RATE: float = 0.05  # ~5% of claims somewhat suspicious


# ---------------- Random seeds ---------------- #
# Granular seeds for internal stochastic processes

SEED_MACRO: int = 10
SEED_POLICYHOLDERS: int = 42
SEED_POLICIES: int = 43
SEED_CLAIMS: int = 44
SEED_EXPOSURE: int = 50

# ---------------- Dataset identity seed ---------------- #
# Used by CLI + manifest to identify a frozen dataset version
# (alias, not a new source of randomness)

SEED_UNIVERSE: int = SEED_POLICYHOLDERS


# ---------------- Anomalies ---------------- #

SEED_ANOMALIES: int = 99
