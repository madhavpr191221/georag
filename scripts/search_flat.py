from __future__ import annotations

import argparse
import json

from georag.retrieval.runner import run_flat_search


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search exact scratch-model embeddings for a held-out EuroSAT tile."
    )
    parser.add_argument("--config", default="configs/milestone_4.toml")
    parser.add_argument("--model", choices=("both", "cnn", "vit"), default="both")
    parser.add_argument("--query-tile-id", default=None)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--metric",
        choices=("cosine", "inner_product", "euclidean"),
        default="cosine",
    )
    arguments = parser.parse_args()
    result = run_flat_search(
        arguments.config,
        model_selection=arguments.model,
        query_tile_id=arguments.query_tile_id,
        k=arguments.k,
        metric=arguments.metric,
    )
    print(json.dumps({"output_directory": result["output_directory"]}, indent=2))


if __name__ == "__main__":
    main()
