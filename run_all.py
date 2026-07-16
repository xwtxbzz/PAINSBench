#!/usr/bin/env python3
"""
PAINSBench: Master pipeline runner.

Usage:
    conda activate dta
    python run_all.py [--steps 1-5]

Steps:
    1. Load & filter activity data, annotate PAINS
    2. Build matched benchmark (PSM)
    3. Generate molecular features
    4. Run baseline experiments + evaluation
    5. (Optional) Adversarial test
"""
import os, sys, argparse, subprocess, time

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")

STEP_SCRIPTS = {
    1: "01_prepare_data.py",
    2: "02_build_benchmark.py",
    3: "03_generate_features.py",
    4: "04_run_experiments.py",
    5: "05_adversarial_test.py",
}

STEP_DESCRIPTIONS = {
    1: "Load & filter activity, annotate PAINS",
    2: "Build matched PAINS+/PAINS- benchmark (PSM)",
    3: "Generate Morgan fingerprints + properties",
    4: "Run baseline experiments + PAINS-aware evaluation",
    5: "Adversarial robustness test (optional)",
}


def run_step(step, dry_run=False):
    script = os.path.join(SCRIPTS_DIR, STEP_SCRIPTS[step])
    desc = STEP_DESCRIPTIONS[step]
    print(f"\n{'#' * 70}")
    print(f"# Step {step}: {desc}")
    print(f"{'#' * 70}\n")

    if dry_run:
        print(f"[DRY RUN] Would execute: python {script}")
        return True

    t0 = time.time()
    ret = subprocess.run([sys.executable, script], cwd=os.path.dirname(__file__))
    elapsed = time.time() - t0
    if ret.returncode != 0:
        print(f"\n❌ Step {step} FAILED (exit code {ret.returncode})")
        return False
    print(f"\n✅ Step {step} completed in {elapsed:.1f}s")
    return True


def main():
    parser = argparse.ArgumentParser(description="PAINSBench: Full pipeline")
    parser.add_argument("--steps", type=str, default="1-4",
                        help="Step range (e.g., 1-4, 3-5, or 1,2,3)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without executing")
    args = parser.parse_args()

    # Parse step range
    if "-" in args.steps:
        parts = args.steps.split("-")
        steps = range(int(parts[0]), int(parts[1]) + 1)
    else:
        steps = [int(s) for s in args.steps.split(",")]

    # Ensure necessary directories exist
    from config import PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR
    for d in [PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR]:
        os.makedirs(d, exist_ok=True)

    # First, pre-extract compound properties if not already done
    props_path = os.path.join(PROCESSED_DIR, "compound_properties.csv")
    if not os.path.exists(props_path) and any(s in [1, 2, 3] for s in steps):
        print("Pre-extracting compound properties...")
        from src.data_utils import load_compound_properties
        props = load_compound_properties()
        props.to_csv(props_path, index=False)
        print(f"Saved: {props_path} ({len(props):,} rows)")

    # Run steps sequentially
    for step in steps:
        if step not in STEP_SCRIPTS:
            print(f"Unknown step: {step}")
            continue
        if not run_step(step, dry_run=args.dry_run):
            sys.exit(1)

    print(f"\n{'=' * 70}")
    print("PAINSBench pipeline complete!")
    print(f"{'=' * 70}")
    print(f"\nResults:  {os.path.join(os.path.dirname(__file__), 'results')}")
    print(f"Figures:  {os.path.join(os.path.dirname(__file__), 'figures')}")


if __name__ == "__main__":
    main()
