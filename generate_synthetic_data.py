#!/usr/bin/env python3
"""
generate_synthetic_data.py — CLI entrypoint for the synthetic, MIMIC-IV-shaped
ICU development data layer (ClinTwin Sentinel, Phase 1).

This produces DEVELOPMENT/ENGINEERING data only. See docs/SYNTHETIC_DATA.md.
All synthetic-data results are engineering/development validation only —
never clinical evidence.

Usage:
    python generate_synthetic_data.py --patients 1000 --seed 42
    python generate_synthetic_data.py --patients 200 --seed 7 --missingness-scale 1.3
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from backend.synthetic_data.generator import GeneratorConfig, generate_cohort, ARCHETYPES, DEFAULT_ARCHETYPE_WEIGHTS
from backend.synthetic_data.extract_to_ml_dataset import extract
from backend.synthetic_data.validate import validate_ml_dataset, dataset_summary


def parse_args():
    p = argparse.ArgumentParser(description="Generate the synthetic MIMIC-IV-shaped ICU development dataset.")
    p.add_argument("--patients", type=int, default=200, help="number of synthetic patients (1 ICU stay each)")
    p.add_argument("--seed", type=int, default=42, help="random seed (full reproducibility)")
    p.add_argument("--min-stay-hours", type=int, default=24)
    p.add_argument("--max-stay-hours", type=int, default=120)
    p.add_argument("--missingness-scale", type=float, default=1.0,
                    help="multiplier applied to all base missingness rates")
    p.add_argument("--outlier-rate", type=float, default=0.01)
    p.add_argument("--output-dir", type=str, default=None,
                    help="defaults to config.SYNTHETIC_DATA_DIR")
    p.add_argument("--archetype-weights", type=str, default=None,
                    help='JSON dict, e.g. \'{"stable":0.3,"improving":0.15,"deteriorating":0.2,"sepsis":0.25,"recovery":0.1}\'')
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = args.output_dir or config.SYNTHETIC_DATA_DIR
    raw_dir = os.path.join(out_dir, "raw")
    processed_dir = os.path.join(out_dir, "processed")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    weights = DEFAULT_ARCHETYPE_WEIGHTS
    if args.archetype_weights:
        weights = json.loads(args.archetype_weights)
        missing = set(ARCHETYPES) - set(weights.keys())
        if missing:
            raise SystemExit(f"--archetype-weights missing keys: {missing}")

    gen_cfg = GeneratorConfig(
        n_patients=args.patients,
        seed=args.seed,
        min_stay_hours=args.min_stay_hours,
        max_stay_hours=args.max_stay_hours,
        missingness_scale=args.missingness_scale,
        outlier_rate=args.outlier_rate,
        archetype_weights=weights,
    )

    print(f"Generating {args.patients} synthetic ICU stays (seed={args.seed})...")
    raw_tables = generate_cohort(gen_cfg)

    for name, df in raw_tables.items():
        path = os.path.join(raw_dir, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"  wrote {path} ({len(df)} rows)")

    print("Extracting flat ml_dataset.csv from raw MIMIC-shaped tables...")
    ml_dataset = extract(raw_tables)
    ml_path = os.path.join(processed_dir, "ml_dataset.csv")
    ml_dataset.to_csv(ml_path, index=False)
    print(f"  wrote {ml_path} ({len(ml_dataset)} rows)")

    print("Validating extracted dataset...")
    report = validate_ml_dataset(ml_dataset)
    print(report)
    if not report.ok:
        print("VALIDATION FAILED — see errors above.", file=sys.stderr)
        sys.exit(1)

    summary = dataset_summary(ml_dataset, icustays=raw_tables["icustays"])
    summary_path = os.path.join(processed_dir, "dataset_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  wrote {summary_path}")
    print(json.dumps(summary, indent=2, default=str))
    print("\nDone. This is SYNTHETIC development data — see docs/SYNTHETIC_DATA.md.")


if __name__ == "__main__":
    main()
