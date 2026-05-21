import re
from pathlib import Path

import jsonlines
from bs4 import BeautifulSoup
from rich.progress import track

from src.utils.logger import logger


class AlpacaFormatter:
    """
    Converts raw scraped data (SO + GitHub) into Alpaca instruction format.
    Synthetic data is already in Alpaca format — passed through with validation.

    Alpaca format:
        instruction: The task description (what the model should do)
        input:       Context (the alert, log snippet, error message)
        output:      The resolution (kubectl commands, explanation, steps)

    HTML tags from SO/GitHub are stripped. Code blocks are preserved as markdown.
    """

    def _strip_html(self, html: str) -> str:
        """
        Strip HTML tags from Stack Overflow API responses.
        Preserves code blocks as markdown fenced blocks.
        BeautifulSoup handles malformed HTML gracefully.
        """
        if not html:
            return ""

        soup = BeautifulSoup(html, "html.parser")

        # Convert <code> and <pre> blocks to markdown before stripping
        for code_block in soup.find_all("pre"):
            code_text = code_block.get_text()
            code_block.replace_with(f"\n```\n{code_text}\n```\n")

        for inline_code in soup.find_all("code"):
            code_text = inline_code.get_text()
            inline_code.replace_with(f"`{code_text}`")

        text = soup.get_text(separator="\n")

        # Collapse multiple blank lines to single
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def format_stackoverflow(self, record: dict) -> dict | None:
        """
        Convert SO Q&A record → Alpaca format.
        instruction = "Given this Stack Overflow question about <tag>, provide a solution."
        input       = question title + cleaned body
        output      = cleaned accepted answer body
        """
        title = record.get("title", "").strip()
        question_body = self._strip_html(record.get("question_body", ""))
        answer_body = self._strip_html(record.get("answer_body", ""))
        tag = record.get("tag", "kubernetes")

        if not title or not answer_body:
            return None

        # Truncate long question bodies — keep first 500 chars as context
        if len(question_body) > 500:
            question_body = question_body[:500] + "..."

        return {
            "instruction": f"You are an expert SRE. Answer this {tag} troubleshooting question with specific commands and steps.",
            "input": f"Question: {title}\n\nContext: {question_body}",
            "output": answer_body,
            "source": "stackoverflow",
            "metadata": {
                "question_id": record.get("question_id"),
                "tag": tag,
                "answer_score": record.get("answer_score", 0),
                "is_accepted": record.get("is_accepted", False),
                "link": record.get("link", ""),
            }
        }

    def format_github_issue(self, record: dict) -> dict | None:
        """
        Convert GitHub Issue record → Alpaca format.
        instruction = "Diagnose and resolve this GitHub issue from <repo>."
        input       = issue title + body (truncated)
        output      = resolution comment
        """
        title = record.get("title", "").strip()
        body = self._strip_html(record.get("body", ""))
        resolution = self._strip_html(record.get("resolution", ""))
        repo = record.get("repo", "")

        if not title or not resolution:
            return None

        # Truncate body — issues can be very long
        if len(body) > 600:
            body = body[:600] + "..."

        repo_short = repo.split("/")[-1] if "/" in repo else repo

        return {
            "instruction": f"You are an expert SRE. Diagnose and resolve this {repo_short} issue with specific commands and steps.",
            "input": f"Issue: {title}\n\nDetails: {body}",
            "output": resolution,
            "source": "github_issues",
            "metadata": {
                "repo": repo,
                "issue_number": record.get("issue_number"),
                "comment_count": record.get("comment_count", 0),
                "url": record.get("url", ""),
            }
        }

    def format_file(self, input_path: Path, source_type: str) -> Path:
        """
        Format an entire raw JSONL file.
        source_type: 'stackoverflow' | 'github_issues' | 'synthetic'
        Returns path to formatted output file.
        """
        output_path = input_path.parent.parent.parent / "processed" / f"{source_type}_formatted.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        records = []
        with jsonlines.open(input_path) as reader:
            records = list(reader)

        formatted = []
        skipped = 0

        for record in track(records, description=f"Formatting {source_type}..."):
            if source_type == "stackoverflow":
                result = self.format_stackoverflow(record)
            elif source_type == "github_issues":
                result = self.format_github_issue(record)
            elif source_type == "synthetic":
                # Synthetic is pre-formatted — validate required fields exist
                result = record if all(k in record for k in ["instruction", "input", "output"]) else None
            else:
                result = None

            if result:
                formatted.append(result)
            else:
                skipped += 1

        with jsonlines.open(output_path, mode="w") as writer:
            for record in formatted:
                writer.write(record)

        logger.info(f"Formatted {source_type}: {len(formatted)} records, {skipped} skipped → {output_path}")
        return output_path
