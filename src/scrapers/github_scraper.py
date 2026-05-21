import os
import time
from pathlib import Path

import jsonlines
from github import Github, GithubException, RateLimitExceededException
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from src.utils.logger import logger

load_dotenv()


class GitHubIssuesScraper:
    """
    Scrapes closed GitHub Issues from core DevOps/SRE repositories.
    Quality signal: closed state + min_comments threshold (no label dependency).
    Closed issues with 3+ comments = resolved problems with discussion thread.
    Output: raw JSONL with issue title + body + resolution comment per line.
    """

    def __init__(self, config: dict):
        self.config = config
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        token = os.getenv("GITHUB_TOKEN")
        if not token:
            raise EnvironmentError("GITHUB_TOKEN not set")
        self.gh = Github(token)

    def _wait_for_rate_limit(self):
        """Proactively wait when remaining requests drop below 100."""
        rate_limit = self.gh.get_rate_limit()
        remaining = rate_limit.core.remaining
        reset_time = rate_limit.core.reset

        if remaining < 100:
            import datetime
            now = datetime.datetime.utcnow()
            wait_seconds = (reset_time - now).total_seconds() + 10
            logger.warning(f"Rate limit low ({remaining} remaining). Waiting {wait_seconds:.0f}s.")
            time.sleep(max(wait_seconds, 0))

    def _extract_resolution(self, issue) -> str:
        """
        Extract resolution from closed issue comments.
        Last comment in a resolved issue is usually the fix confirmation.
        Falls back to highest-reaction comment if last is too short.
        """
        try:
            comments = list(issue.get_comments())
            if not comments:
                return ""

            last_comment = comments[-1].body or ""

            # If last comment is too short, find the most reacted comment
            if len(last_comment) < 100 and len(comments) > 1:
                best = max(
                    comments,
                    key=lambda c: getattr(c, 'reactions', {}).get('total_count', 0)
                )
                return best.body or last_comment

            return last_comment

        except Exception as e:
            logger.debug(f"Could not extract resolution: {e}")
            return ""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=5, max=60),
        retry=retry_if_exception_type(Exception),
        reraise=False
    )
    def scrape_repo(self, repo_name: str) -> list[dict]:
        """
        Scrape closed issues from one repository.
        No label filter — rely on min_comments for quality signal.
        """
        self._wait_for_rate_limit()
        logger.info(f"Scraping repo: {repo_name}")

        try:
            repo = self.gh.get_repo(repo_name)
        except GithubException as e:
            logger.error(f"Cannot access repo {repo_name}: {e}")
            return []

        filters = self.config["filters"]
        max_issues = filters.get("max_issues_per_repo", 100)
        min_comments = filters.get("min_comments", 3)

        records = []
        count = 0

        # No labels param — fetch all closed issues sorted by most commented
        issues = repo.get_issues(
            state="closed",
            sort="comments",
            direction="desc",
        )

        for issue in issues:
            if count >= max_issues:
                break

            # Skip pull requests — GitHub API returns PRs under issues endpoint
            if issue.pull_request:
                continue

            if issue.comments < min_comments:
                # Issues are sorted by comments desc — once below threshold we're done
                break

            resolution = self._extract_resolution(issue)
            if not resolution or len(resolution) < 50:
                continue

            records.append({
                "source": "github_issues",
                "repo": repo_name,
                "issue_number": issue.number,
                "title": issue.title or "",
                "body": issue.body or "",
                "resolution": resolution,
                "labels": [l.name for l in issue.labels],
                "comment_count": issue.comments,
                "url": issue.html_url,
                "created_at": issue.created_at.isoformat() if issue.created_at else "",
                "closed_at": issue.closed_at.isoformat() if issue.closed_at else "",
            })

            count += 1
            time.sleep(0.3)

        logger.info(f"  {len(records)} quality issues from {repo_name}")
        return records

    def scrape_all(self) -> Path:
        """Scrape all configured repositories, write to JSONL."""
        output_file = self.output_dir / "github_issues_raw.jsonl"
        repos = self.config["repositories"]
        total_records = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total} repos"),
        ) as progress:
            task = progress.add_task("Scraping GitHub Issues...", total=len(repos))

            with jsonlines.open(output_file, mode="w") as writer:
                for repo_name in repos:
                    try:
                        records = self.scrape_repo(repo_name)
                        for record in records:
                            writer.write(record)
                        total_records += len(records)
                    except Exception as e:
                        logger.error(f"Failed to scrape {repo_name}: {e}")
                    finally:
                        progress.advance(task)

        logger.info(f"GitHub scrape complete: {total_records} records → {output_file}")
        return output_file
