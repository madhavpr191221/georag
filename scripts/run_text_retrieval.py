"""Run exact natural-language-to-image retrieval with RemoteCLIP."""

from __future__ import annotations

import argparse

from georag.experiments.text_retrieval import run_text_retrieval


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/milestone_7.toml")
    parser.add_argument("--device", default="auto", help="auto, cpu, or a torch device such as cuda:0")
    parser.add_argument("--max-gallery", type=int, help="encode only the first N train tiles for a smoke run")
    parser.add_argument("--output-dir", help="optional output directory override")
    args = parser.parse_args()
    run_text_retrieval(
        args.config,
        device=args.device,
        max_gallery=args.max_gallery,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
