"""Run the OpenAI vision-assisted review of M7 retrieved examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tomllib

from dotenv import load_dotenv

from georag.experiments.vlm_audit import DEFAULT_MODEL, run_vlm_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/milestone_7.toml")
    parser.add_argument("--audit-csv", default="experiments/milestone_7/human_audit.csv")
    parser.add_argument("--output-dir", default="experiments/milestone_7/vlm_audit")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, help="judge only the first N rows (useful for a small paid smoke run)")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    repo_root = config_path.parents[1]
    load_dotenv(repo_root / ".env", override=False)
    with config_path.open("rb") as source:
        config = tomllib.load(source)
    data_root = Path(config["dataset"]["root"])
    if not data_root.is_absolute():
        data_root = repo_root / data_root
    audit_csv = Path(args.audit_csv)
    if not audit_csv.is_absolute():
        audit_csv = repo_root / audit_csv
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir

    summary = run_vlm_audit(
        audit_csv,
        data_root,
        output_dir,
        model=args.model,
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nReport: {output_dir / 'report.md'}")
    print(f"Gallery: {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
