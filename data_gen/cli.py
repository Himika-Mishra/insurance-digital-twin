"""
Command-line entry point for generating a locked synthetic insurance universe.

Usage (from project root):

    python -m data_gen.cli

This script:
1) Generates a deterministic synthetic insurance dataset
2) Writes CSV snapshots to disk
3) Produces a cryptographic dataset manifest to lock the snapshot
"""

from pathlib import Path
import json
import hashlib
import datetime
import platform

from .generators import generate_universe
from . import config


# -------------------------------------------------------------------
# Output location
# -------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def file_hash(path: Path) -> str:
    """Compute SHA-256 hash of a file (streaming-safe)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# -------------------------------------------------------------------
# Main entrypoint
# -------------------------------------------------------------------

def main() -> None:
    print("▶ Generating synthetic insurance universe...")

    macro, holders, policies, claims = generate_universe()

    # ---------------- Write datasets ---------------- #

    paths = {
        "macro.csv": DATA_DIR / "macro.csv",
        "policyholders.csv": DATA_DIR / "policyholders.csv",
        "policies.csv": DATA_DIR / "policies.csv",
        "claims.csv": DATA_DIR / "claims.csv",
    }

    macro.to_csv(paths["macro.csv"], index=False)
    holders.to_csv(paths["policyholders.csv"], index=False)
    policies.to_csv(paths["policies.csv"], index=False)
    claims.to_csv(paths["claims.csv"], index=False)

    print(f"✔ Data written to {DATA_DIR}")

    # ---------------- Build dataset manifest ---------------- #

    manifest = {
        "dataset_version": "v1.0",
        "generated_at_utc": datetime.datetime.utcnow().isoformat(),
        "generator_entrypoint": "data_gen.cli",
        "generator_function": "generate_universe",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "seeds": {
            "universe": config.SEED_UNIVERSE,
            "claims": config.SEED_CLAIMS,
            "anomalies": config.SEED_ANOMALIES,
        },
        "row_counts": {
            "macro": len(macro),
            "policyholders": len(holders),
            "policies": len(policies),
            "claims": len(claims),
        },
        "file_hashes_sha256": {
            name: file_hash(path) for name, path in paths.items()
        },
    }

    manifest_path = DATA_DIR / "dataset_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("✔ Dataset manifest written")
    print(f"✔ Manifest path: {manifest_path}")

    # ---------------- Quick portfolio sanity ---------------- #

    if len(claims) > 0:
        lr = claims["paid_amount"].sum() / policies["base_annual_premium"].sum()
        print(f"ℹ Approx portfolio loss ratio (paid / premium): {lr:.2%}")

    print("✅ Dataset generation complete and LOCKED")


# -------------------------------------------------------------------
# CLI hook
# -------------------------------------------------------------------

if __name__ == "__main__":
    main()
