import os
import json
import time
from pathlib import Path
from typing import Optional

import jsonlines
from stackapi import StackAPI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from src.utils.logger import logger

load_dotenv()


class StackOverflowScraper:
    """
    Scrapes Stack Overflow Q&A pairs using the StackExchange API.
    Targets DevOps/SRE tags: kubernetes, prometheus, terraform, helm, etc.
    Filters for high-quality: accepted answers, min score threshold.
    Output: raw JSONL with question + accepted answer per line.
    """

    def __init__(self, config: dict):
        self.config = config
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Init StackAPI — handles pagination and rate limiting internally
        api_key = os.getenv("STACKOVERFLOW_API_KEY")
        self.api = StackAPI("stackoverflow", key=api_key)
        self.api.page_size = config["filters"]["pagesize"]
        self.api.max_pages = config["filters"]["max_pages"]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def _fetch_questions(self, tag: str) -> list[dict]:
        """
        Fetch questions for a single tag with retry + exponential backoff.
        StackAPI handles quota — 300 req/day free, 10k/day with key.
        """
        logger.info(f"Fetching questions for tag: {tag}")
        questions = self.api.fetch(
            "questions",
            tagged=tag,
            filter="withbody",          # include question body
            sort="votes",
            order="desc",
            min=self.config["filters"]["min_score"],
        )
        return questions.get("items", [])

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def _fetch_answers(self, question_ids: list[int]) -> dict[int, dict]:
        """
        Batch fetch accepted answers for a list of question IDs.
        Returns dict: question_id → answer dict.
        StackAPI accepts semicolon-joined IDs for batch calls — reduces quota usage.
        """
        if not question_ids:
            return {}

        # Batch in groups of 100 — StackAPI limit per request
        answers_map = {}
        for i in range(0, len(question_ids), 100):
            batch = question_ids[i:i+100]
            ids_str = ";".join(str(qid) for qid in batch)

            result = self.api.fetch(
                f"questions/{ids_str}/answers",
                filter="withbody",
                sort="votes",
                order="desc",
            )

            for answer in result.get("items", []):
                qid = answer.get("question_id")
                # Keep only the top-voted answer per question
                if qid not in answers_map:
                    answers_map[qid] = answer

            time.sleep(0.5)  # Polite delay between batch calls

        return answers_map

    def _is_quality_answer(self, answer: dict) -> bool:
        """
        Quality gate: accepted answer OR score >= 5.
        Filters out low-signal, unverified answers.
        """
        return (
            answer.get("is_accepted", False) or
            answer.get("score", 0) >= 5
        )

    def scrape_tag(self, tag: str) -> list[dict]:
        """
        Full scrape for one tag: fetch questions → fetch answers → join → filter.
        Returns list of raw QA dicts ready for formatting.
        """
        questions = self._fetch_questions(tag)
        logger.info(f"  Got {len(questions)} questions for [{tag}]")

        # Extract IDs of questions that have accepted answers
        question_ids = [
            q["question_id"] for q in questions
            if q.get("answer_count", 0) > 0
        ]

        answers_map = self._fetch_answers(question_ids)
        logger.info(f"  Fetched {len(answers_map)} answers for [{tag}]")

        # Join question + answer, apply quality filter
        records = []
        for q in questions:
            qid = q.get("question_id")
            answer = answers_map.get(qid)

            if not answer or not self._is_quality_answer(answer):
                continue

            records.append({
                "source": "stackoverflow",
                "tag": tag,
                "question_id": qid,
                "title": q.get("title", ""),
                "question_body": q.get("body", ""),
                "answer_body": answer.get("body", ""),
                "question_score": q.get("score", 0),
                "answer_score": answer.get("score", 0),
                "is_accepted": answer.get("is_accepted", False),
                "link": q.get("link", ""),
                "tags": q.get("tags", []),
            })

        logger.info(f"  {len(records)} quality pairs retained for [{tag}]")
        return records

    def scrape_all(self) -> Path:
        """
        Scrape all configured tags, write to JSONL.
        Returns path to output file.
        """
        output_file = self.output_dir / "stackoverflow_raw.jsonl"
        tags = self.config["tags"]
        total_records = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total} tags"),
        ) as progress:
            task = progress.add_task("Scraping Stack Overflow...", total=len(tags))

            with jsonlines.open(output_file, mode="w") as writer:
                for tag in tags:
                    try:
                        records = self.scrape_tag(tag)
                        for record in records:
                            writer.write(record)
                        total_records += len(records)
                        time.sleep(1)  # Respect rate limit between tags
                    except Exception as e:
                        logger.error(f"Failed to scrape tag [{tag}]: {e}")
                    finally:
                        progress.advance(task)

        logger.info(f"Stack Overflow scrape complete: {total_records} records → {output_file}")
        return output_file
