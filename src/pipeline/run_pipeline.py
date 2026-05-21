"""
P1 Pipeline Orchestrator
Stages: scrape → format → validate → upload
Run: python -m src.pipeline.run_pipeline --stage <stage>
"""

import os
import click
import yaml
from pathlib import Path
from dotenv import load_dotenv

from src.utils.logger import logger

load_dotenv()

with open("configs/scraper_config.yaml") as f:
    SCRAPER_CONFIG = yaml.safe_load(f)

with open("configs/dataset_config.yaml") as f:
    DATASET_CONFIG = yaml.safe_load(f)


@click.command()
@click.option("--stage", type=click.Choice(["scrape", "format", "validate", "upload", "all"]), default="all")
def main(stage: str):

    if stage in ("scrape", "all"):
        logger.info("=== Stage 1: Scraping ===")

        # Stack Overflow — only if raw file doesn't already exist (resume-safe)
        so_output = Path("data/raw/stackoverflow/stackoverflow_raw.jsonl")
        if so_output.exists():
            logger.info(f"SO data already exists ({so_output}) — skipping scrape. Delete file to re-scrape.")
        else:
            from src.scrapers.stackoverflow_scraper import StackOverflowScraper
            so_scraper = StackOverflowScraper(SCRAPER_CONFIG["stackoverflow"])
            so_scraper.scrape_all()

        # GitHub — skip gracefully if token not set
        gh_token = os.getenv("GITHUB_TOKEN")
        if not gh_token:
            logger.warning("GITHUB_TOKEN not set — skipping GitHub scraper. Add token to .env to enable.")
        else:
            gh_output = Path("data/raw/github_issues/github_issues_raw.jsonl")
            if gh_output.exists():
                logger.info(f"GitHub data already exists ({gh_output}) — skipping scrape.")
            else:
                from src.scrapers.github_scraper import GitHubIssuesScraper
                gh_scraper = GitHubIssuesScraper(SCRAPER_CONFIG["github"])
                gh_scraper.scrape_all()

        # Synthetic — always runs, purely local
        syn_output = Path("data/raw/synthetic/synthetic_incidents.jsonl")
        if syn_output.exists():
            logger.info(f"Synthetic data already exists ({syn_output}) — skipping generation.")
        else:
            from src.scrapers.synthetic_generator import SyntheticIncidentGenerator
            syn_gen = SyntheticIncidentGenerator(SCRAPER_CONFIG["synthetic"])
            syn_gen.generate_all()

    if stage in ("format", "all"):
        logger.info("=== Stage 2: Formatting ===")
        from src.formatters.alpaca_formatter import AlpacaFormatter
        formatter = AlpacaFormatter()

        raw_dir = Path("data/raw")

        # Format each source only if raw file exists
        source_map = {
            "stackoverflow": raw_dir / "stackoverflow/stackoverflow_raw.jsonl",
            "github_issues": raw_dir / "github_issues/github_issues_raw.jsonl",
            "synthetic":     raw_dir / "synthetic/synthetic_incidents.jsonl",
        }

        for source_type, raw_path in source_map.items():
            if raw_path.exists():
                formatter.format_file(raw_path, source_type)
            else:
                logger.warning(f"Raw file not found for {source_type} — skipping format stage for this source.")

    if stage in ("validate", "all"):
        logger.info("=== Stage 3: Validating ===")
        from src.validators.dataset_validator import DatasetValidator
        validator = DatasetValidator(DATASET_CONFIG)

        # Only validate files that actually exist
        all_formatted = [
            Path("data/processed/stackoverflow_formatted.jsonl"),
            Path("data/processed/github_issues_formatted.jsonl"),
            Path("data/processed/synthetic_formatted.jsonl"),
        ]
        formatted_files = [f for f in all_formatted if f.exists()]

        if not formatted_files:
            logger.error("No formatted files found. Run --stage format first.")
            return

        logger.info(f"Validating {len(formatted_files)} formatted files: {[f.name for f in formatted_files]}")
        validator.validate_and_save(formatted_files)

    if stage in ("upload", "all"):
        logger.info("=== Stage 4: Uploading to S3 ===")
        from src.utils.s3_uploader import S3Uploader
        uploader = S3Uploader()

        validated_files = {
            "train":      Path("data/validated/sre_ops_train_v1.jsonl"),
            "validation": Path("data/validated/sre_ops_val_v1.jsonl"),
            "test":       Path("data/validated/sre_ops_test_v1.jsonl"),
        }
        uris = uploader.upload_dataset(validated_files)
        for split, uri in uris.items():
            logger.info(f"  {split}: {uri}")


if __name__ == "__main__":
    main()
