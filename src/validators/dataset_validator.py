import hashlib
import random
from pathlib import Path
from collections import defaultdict

import jsonlines
from pydantic import BaseModel, field_validator, ValidationError
from rich.console import Console
from rich.table import Table

from src.utils.logger import logger

console = Console()


class AlpacaSample(BaseModel):
    """
    Pydantic schema for a single Alpaca training sample.
    Validation runs on every record — catches schema drift early.
    """
    instruction: str
    input: str
    output: str
    source: str = "unknown"
    metadata: dict = {}

    @field_validator("instruction")
    @classmethod
    def instruction_not_empty(cls, v):
        if len(v.strip()) < 20:
            raise ValueError(f"Instruction too short: {len(v)} chars (min 20)")
        return v.strip()

    @field_validator("output")
    @classmethod
    def output_quality(cls, v):
        if len(v.strip()) < 50:
            raise ValueError(f"Output too short: {len(v)} chars (min 50)")
        if len(v.split()) < 15:
            raise ValueError(f"Output too few words: {len(v.split())} (min 15)")
        return v.strip()


class DatasetValidator:
    """
    Quality gate for the processed dataset.
    Runs three checks:
      1. Schema validation (Pydantic) — catches missing/malformed fields
      2. Deduplication — removes near-identical samples (Jaccard on token sets)
      3. Split generation — train/val/test stratified by source
    Produces final validated JSONL files ready for fine-tuning.
    """

    def __init__(self, config: dict):
        self.config = config
        self.quality = config["quality"]
        self.splits = config["splits"]
        self.output_dir = Path(config["output"]["validated_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """
        Token-level Jaccard similarity between two texts.
        Used for near-duplicate detection — catches rephrased duplicates.
        Jaccard = |intersection| / |union| of token sets.
        """
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        if not tokens1 or not tokens2:
            return 0.0
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        return len(intersection) / len(union)

    def _content_hash(self, record: dict) -> str:
        """MD5 hash of instruction+output for exact duplicate detection."""
        content = record["instruction"] + record["output"]
        return hashlib.md5(content.encode()).hexdigest()

    def validate_and_deduplicate(self, input_files: list[Path]) -> list[dict]:
        """
        Load all formatted files, validate schema, deduplicate.
        Returns clean list of validated records.
        """
        all_records = []

        # Load all formatted files
        for file_path in input_files:
            if not file_path.exists():
                logger.warning(f"File not found, skipping: {file_path}")
                continue
            with jsonlines.open(file_path) as reader:
                for record in reader:
                    all_records.append(record)

        logger.info(f"Loaded {len(all_records)} total records from {len(input_files)} files")

        # Step 1: Schema validation
        valid_records = []
        schema_failures = 0

        for record in all_records:
            try:
                validated = AlpacaSample(**record)
                valid_records.append(validated.model_dump())
            except ValidationError as e:
                schema_failures += 1
                logger.debug(f"Schema validation failed: {e.errors()[0]['msg']}")

        logger.info(f"Schema validation: {len(valid_records)} passed, {schema_failures} failed")

        # Step 2: Exact deduplication via MD5 hash
        seen_hashes = set()
        deduped = []
        exact_dupes = 0

        for record in valid_records:
            h = self._content_hash(record)
            if h not in seen_hashes:
                seen_hashes.add(h)
                deduped.append(record)
            else:
                exact_dupes += 1

        logger.info(f"Exact dedup: removed {exact_dupes} duplicates, {len(deduped)} remaining")

        # Step 3: Near-duplicate removal (Jaccard)
        # O(n²) — acceptable for <10k samples, use MinHash for larger datasets
        threshold = self.quality["dedup_threshold"]
        final_records = []
        near_dupes = 0

        for i, record in enumerate(deduped):
            is_near_dupe = False
            for j in range(max(0, i - 50), i):  # Check against last 50 only — sliding window
                sim = self._jaccard_similarity(
                    deduped[j]["instruction"] + deduped[j]["output"],
                    record["instruction"] + record["output"]
                )
                if sim > threshold:
                    is_near_dupe = True
                    near_dupes += 1
                    break

            if not is_near_dupe:
                final_records.append(record)

        logger.info(f"Near-dupe removal (Jaccard>{threshold}): removed {near_dupes}, {len(final_records)} remaining")
        return final_records

    def generate_splits(self, records: list[dict]) -> dict[str, list[dict]]:
        """
        Stratified train/val/test split by source.
        Stratification ensures each split has representation from SO, GitHub, synthetic.
        """
        # Group by source
        by_source = defaultdict(list)
        for record in records:
            by_source[record.get("source", "unknown")].append(record)

        splits = {"train": [], "validation": [], "test": []}

        for source, source_records in by_source.items():
            random.shuffle(source_records)
            n = len(source_records)

            train_end = int(n * self.splits["train"])
            val_end = train_end + int(n * self.splits["validation"])

            splits["train"].extend(source_records[:train_end])
            splits["validation"].extend(source_records[train_end:val_end])
            splits["test"].extend(source_records[val_end:])

        # Shuffle each split
        for split in splits.values():
            random.shuffle(split)

        return splits

    def validate_and_save(self, input_files: list[Path]) -> dict[str, Path]:
        """
        Full validation pipeline: load → validate → deduplicate → split → save.
        Returns dict of split_name → output_path.
        """
        clean_records = self.validate_and_deduplicate(input_files)

        if len(clean_records) < 100:
            logger.warning(f"Only {len(clean_records)} records after validation — dataset may be too small for fine-tuning")

        splits = self.generate_splits(clean_records)

        output_files = {}
        file_map = {
            "train": self.config["output"]["train_file"],
            "validation": self.config["output"]["val_file"],
            "test": self.config["output"]["test_file"],
        }

        for split_name, split_records in splits.items():
            output_path = self.output_dir / file_map[split_name]
            with jsonlines.open(output_path, mode="w") as writer:
                for record in split_records:
                    writer.write(record)
            output_files[split_name] = output_path
            logger.info(f"Saved {split_name}: {len(split_records)} records → {output_path}")

        # Print summary table
        self._print_summary(splits, clean_records)
        return output_files

    def _print_summary(self, splits: dict, all_records: list):
        """Rich table summary of final dataset stats."""
        table = Table(title="Dataset Validation Summary", show_header=True)
        table.add_column("Split", style="cyan")
        table.add_column("Records", justify="right")
        table.add_column("SO", justify="right")
        table.add_column("GitHub", justify="right")
        table.add_column("Synthetic", justify="right")

        for split_name, records in splits.items():
            so = sum(1 for r in records if r.get("source") == "stackoverflow")
            gh = sum(1 for r in records if r.get("source") == "github_issues")
            syn = sum(1 for r in records if r.get("source") == "synthetic")
            table.add_row(split_name, str(len(records)), str(so), str(gh), str(syn))

        console.print(table)
        console.print(f"[green]Total clean records: {len(all_records)}[/green]")
