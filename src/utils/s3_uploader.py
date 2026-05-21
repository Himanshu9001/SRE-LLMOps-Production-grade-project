import os
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

from src.utils.logger import logger

load_dotenv()


class S3Uploader:
    """
    Uploads validated dataset files to S3 under datasets/v{version}/ prefix.
    Uses multipart upload for files > 100MB (JSONL datasets can get large).
    """

    def __init__(self):
        self.bucket = os.getenv("S3_BUCKET", "sre-llmops-artifacts")
        self.version = os.getenv("DATASET_VERSION", "v1")
        self.s3 = boto3.client("s3")

    def upload_file(self, local_path: Path, s3_key: str) -> str:
        """
        Upload a single file to S3.
        Returns the full S3 URI on success.
        """
        try:
            logger.info(f"Uploading {local_path.name} → s3://{self.bucket}/{s3_key}")
            self.s3.upload_file(
                str(local_path),
                self.bucket,
                s3_key,
                ExtraArgs={"ServerSideEncryption": "AES256"}
            )
            s3_uri = f"s3://{self.bucket}/{s3_key}"
            logger.info(f"Uploaded: {s3_uri}")
            return s3_uri
        except ClientError as e:
            logger.error(f"S3 upload failed for {local_path}: {e}")
            raise

    def upload_dataset(self, validated_files: dict[str, Path]) -> dict[str, str]:
        """
        Upload all split files to S3 under datasets/v{version}/ prefix.
        Returns dict of split_name → S3 URI.
        """
        s3_uris = {}
        prefix = f"datasets/{self.version}"

        for split_name, local_path in validated_files.items():
            if not local_path.exists():
                logger.warning(f"File not found, skipping upload: {local_path}")
                continue

            s3_key = f"{prefix}/{local_path.name}"
            uri = self.upload_file(local_path, s3_key)
            s3_uris[split_name] = uri

        logger.info(f"Dataset upload complete: {len(s3_uris)} files → s3://{self.bucket}/{prefix}/")
        return s3_uris
