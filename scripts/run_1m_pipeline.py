#!/usr/bin/env python3
"""CLI entrypoint for running the ChessLab streaming data ingestion pipeline."""

import argparse
import json
from chesslab.data.pipeline import PipelineConfig, run_ingestion_pipeline


def main():
    parser = argparse.ArgumentParser(description="ChessLab streaming data pipeline")
    parser.add_argument(
        "--source",
        type=str,
        default="https://database.lichess.org/standard/lichess_db_standard_rated_2024-01.pgn.zst",
        help="HTTP URL or local path to .pgn.zst archive",
    )
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory")
    parser.add_argument("--max-games", type=int, default=10000, help="Target valid games (default: 10,000)")
    parser.add_argument("--report-every", type=int, default=1000, help="Progress report interval")
    parser.add_argument("--min-elo", type=int, default=1500, help="Minimum Elo for both players")
    args = parser.parse_args()

    cfg = PipelineConfig(
        source=args.source,
        output_dir=args.output_dir,
        max_valid_games=args.max_games,
        report_every=args.report_every,
        min_elo=args.min_elo,
    )

    manifest = run_ingestion_pipeline(cfg)

    print("\nPipeline execution completed successfully.")
    print(f"Manifest written to: {args.output_dir}/manifests/dataset_manifest.json")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
