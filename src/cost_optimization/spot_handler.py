"""
P12b — Spot Instance Interruption Handler
Handles AWS spot interruptions gracefully during training.

Spot interruption sequence:
  T-120s: AWS sends interruption notice via EC2 metadata API
           IMDS: GET /latest/meta-data/spot/interruption-action
  T-0:    Instance terminated

Response strategy:
  T-120s to T-90s: Detect interruption notice
  T-90s to T-60s:  Save checkpoint to S3
  T-60s to T-30s:  Clean up temp files, flush metrics to MLflow
  T-30s to T-0s:   Graceful pod termination

Kubernetes integration:
  Spot interruption notice → node taint added
  Kubernetes drains node (30s grace period)
  Pod receives SIGTERM → training saves checkpoint
  Pod terminated after terminationGracePeriodSeconds

Training resume:
  New GPU node starts → downloads checkpoint from S3
  Training resumes from last checkpoint (not from scratch)
  Max wasted compute: 1 checkpoint interval (default: 100 steps)
"""

import os
import time
import signal
import threading
import requests
import boto3
from pathlib import Path
from src.utils.logger import logger


IMDS_TOKEN_URL    = "http://169.254.169.254/latest/api/token"
IMDS_SPOT_URL     = "http://169.254.169.254/latest/meta-data/spot/interruption-action"
IMDS_INSTANCE_URL = "http://169.254.169.254/latest/meta-data/instance-id"
IMDS_TTL          = 21600  # 6 hours


def get_imds_token() -> str:
    """Get IMDSv2 token for metadata requests."""
    response = requests.put(
        IMDS_TOKEN_URL,
        headers={"X-aws-ec2-metadata-token-ttl-seconds": str(IMDS_TTL)},
        timeout=1,
    )
    return response.text


def check_spot_interruption() -> bool:
    """
    Check if spot interruption notice received.
    Returns True if instance will be interrupted in ~2 minutes.
    Called every 5 seconds during training.
    """
    try:
        token    = get_imds_token()
        response = requests.get(
            IMDS_SPOT_URL,
            headers={"X-aws-ec2-metadata-token": token},
            timeout=1,
        )
        # 200 = interruption notice present
        # 404 = no interruption (normal state)
        return response.status_code == 200

    except requests.exceptions.RequestException:
        # Can't reach IMDS — not on EC2 or network issue
        return False


class SpotInterruptionHandler:
    """
    Background thread that polls for spot interruption notices.
    Triggers graceful checkpoint save when notice detected.

    Usage in training loop:
        handler = SpotInterruptionHandler(
            checkpoint_fn=lambda: trainer.save_checkpoint(),
            s3_bucket="sre-llmops-artifacts",
            s3_prefix="checkpoints/interrupted",
        )
        handler.start()

        # Training loop runs normally
        # If spot interrupted, handler saves checkpoint automatically

        handler.stop()
    """

    def __init__(
        self,
        checkpoint_fn,    # callable that saves current state
        s3_bucket: str,
        s3_prefix: str,
        poll_interval: float = 5.0,   # check every 5 seconds
    ):
        self.checkpoint_fn  = checkpoint_fn
        self.s3_bucket      = s3_bucket
        self.s3_prefix      = s3_prefix
        self.poll_interval  = poll_interval
        self._thread        = None
        self._running       = False
        self._interrupted   = False

        # Register SIGTERM handler (Kubernetes sends this on pod deletion)
        signal.signal(signal.SIGTERM, self._handle_sigterm)
        signal.signal(signal.SIGINT,  self._handle_sigterm)

    def _handle_sigterm(self, signum, frame):
        """
        Handle SIGTERM from Kubernetes graceful shutdown.
        Kubernetes sends SIGTERM → waits terminationGracePeriodSeconds → SIGKILL
        We have that window to save checkpoint.
        """
        logger.warning(f"Received signal {signum} — saving checkpoint before shutdown")
        self._save_checkpoint_and_upload(reason="sigterm")
        self._running = False

    def _save_checkpoint_and_upload(self, reason: str = "interruption"):
        """Save checkpoint locally then upload to S3."""
        checkpoint_dir = f"/tmp/checkpoint-{reason}-{int(time.time())}"

        try:
            logger.info(f"Saving checkpoint due to {reason}...")
            self.checkpoint_fn(checkpoint_dir)
            logger.info(f"Checkpoint saved to {checkpoint_dir}")

            # Upload to S3
            s3  = boto3.client("s3", region_name="us-east-1")
            s3_prefix = f"{self.s3_prefix}/{reason}"

            for f in Path(checkpoint_dir).rglob("*"):
                if f.is_file():
                    s3_key = f"{s3_prefix}/{f.relative_to(checkpoint_dir)}"
                    s3.upload_file(str(f), self.s3_bucket, s3_key)

            logger.info(f"Checkpoint uploaded: s3://{self.s3_bucket}/{s3_prefix}/")
            self._interrupted = True

        except Exception as e:
            logger.error(f"Checkpoint save failed: {e}")

    def start(self):
        """Start background interruption monitor thread."""
        self._running = True
        self._thread  = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
        )
        self._thread.start()
        logger.info("Spot interruption handler started")

    def stop(self):
        """Stop monitoring thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _monitor_loop(self):
        """Poll IMDS every poll_interval seconds."""
        while self._running:
            if check_spot_interruption():
                logger.warning("SPOT INTERRUPTION NOTICE RECEIVED — saving checkpoint")
                self._save_checkpoint_and_upload(reason="spot-interruption")
                break  # stop monitoring after handling
            time.sleep(self.poll_interval)

    @property
    def was_interrupted(self) -> bool:
        """True if interruption was detected and handled."""
        return self._interrupted


class CheckpointManager:
    """
    Manages training checkpoints with S3 versioning.
    Enables resume from any previous checkpoint.

    Checkpoint strategy:
      Save every N steps (default: 100)
      Keep last K checkpoints (default: 3) — older ones deleted from S3
      On interruption: save immediately regardless of step count

    Resume logic:
      1. Check S3 for latest checkpoint
      2. Download to local /tmp
      3. Load into model/optimizer state
      4. Continue from that step
    """

    def __init__(
        self,
        s3_bucket: str,
        s3_prefix: str,
        keep_last: int = 3,
        region: str = "us-east-1",
    ):
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.keep_last = keep_last
        self.s3        = boto3.client("s3", region_name=region)

    def save(
        self,
        model,
        optimizer,
        step: int,
        epoch: int,
        loss: float,
        local_dir: str = "/tmp/checkpoint",
    ) -> str:
        """Save checkpoint locally and upload to S3."""
        checkpoint_key = f"{self.s3_prefix}/step-{step:06d}"
        local_path     = Path(local_dir) / f"step-{step:06d}"
        local_path.mkdir(parents=True, exist_ok=True)

        # Save model adapter
        model.save_pretrained(str(local_path))

        # Save training state
        import torch
        torch.save({
            "step":       step,
            "epoch":      epoch,
            "loss":       loss,
            "optimizer":  optimizer.state_dict(),
        }, str(local_path / "training_state.pt"))

        # Upload to S3
        for f in local_path.rglob("*"):
            if f.is_file():
                s3_key = f"{checkpoint_key}/{f.relative_to(local_path)}"
                self.s3.upload_file(str(f), self.s3_bucket, s3_key)

        logger.info(f"Checkpoint saved: s3://{self.s3_bucket}/{checkpoint_key}/")

        # Clean up old checkpoints
        self._cleanup_old_checkpoints()

        return checkpoint_key

    def get_latest(self) -> Optional[str]:
        """Get S3 key of the latest checkpoint, if any."""
        response = self.s3.list_objects_v2(
            Bucket=self.s3_bucket,
            Prefix=f"{self.s3_prefix}/step-",
            Delimiter="/",
        )

        prefixes = [p["Prefix"] for p in response.get("CommonPrefixes", [])]
        if not prefixes:
            return None

        # Sort by step number — latest is last
        return sorted(prefixes)[-1].rstrip("/")

    def download(self, s3_prefix: str, local_dir: str = "/tmp/checkpoint") -> Path:
        """Download checkpoint from S3."""
        local_path = Path(local_dir)
        local_path.mkdir(parents=True, exist_ok=True)

        response = self.s3.list_objects_v2(
            Bucket=self.s3_bucket,
            Prefix=s3_prefix,
        )

        for obj in response.get("Contents", []):
            key       = obj["Key"]
            relative  = key[len(s3_prefix):].lstrip("/")
            local_file = local_path / relative
            local_file.parent.mkdir(parents=True, exist_ok=True)
            self.s3.download_file(self.s3_bucket, key, str(local_file))

        logger.info(f"Checkpoint downloaded to {local_path}")
        return local_path

    def _cleanup_old_checkpoints(self):
        """Delete old checkpoints, keeping only last K."""
        response = self.s3.list_objects_v2(
            Bucket=self.s3_bucket,
            Prefix=f"{self.s3_prefix}/step-",
            Delimiter="/",
        )

        prefixes = sorted(
            p["Prefix"] for p in response.get("CommonPrefixes", [])
        )

        # Delete all but last keep_last
        to_delete = prefixes[:-self.keep_last]
        for prefix in to_delete:
            objects = self.s3.list_objects_v2(
                Bucket=self.s3_bucket, Prefix=prefix
            )
            for obj in objects.get("Contents", []):
                self.s3.delete_object(Bucket=self.s3_bucket, Key=obj["Key"])

            logger.debug(f"Deleted old checkpoint: {prefix}")


# Make Optional available
from typing import Optional
