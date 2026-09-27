"""Run the evaluation suite over baselines/ablations.

  python scripts/benchmark.py --profile fixture --variants full no_critic no_ask_user no_vlm fixed_pipeline no_pagerank
  python scripts/benchmark.py --profile workstation --manifest data/eval/real_cases.jsonl --variants full fixed_pipeline

Outputs runs/benchmarks/<timestamp>/results.{json,md}.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from personal_angel.evaluation import VARIANTS, run_suite, to_markdown

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="fixture")
    parser.add_argument("--manifest", default="data/eval/synthetic_cases.jsonl")
    parser.add_argument("--variants", nargs="+", default=["full", "no_critic", "no_ask_user", "no_vlm", "fixed_pipeline"])
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    unknown = [v for v in args.variants if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants {unknown}; choose from {list(VARIANTS)}")
    out = Path(args.out or ROOT / "runs" / "benchmarks" / time.strftime("%Y%m%d-%H%M%S"))
    result = run_suite(ROOT / args.manifest, args.profile, args.variants, out, with_judge=not args.no_judge)
    print("\n" + to_markdown(result["summary"]))
    print(f"written to {out}")

if __name__ == "__main__":
    main()
