"""
Background certificate processing queue system with Redis persistence.

This module provides a Redis-backed worker queue for processing certificates
asynchronously without blocking the main application. Tasks survive server restarts.
"""

import os
import logging
import threading
import uuid
import time
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

import redis
from redis.exceptions import ConnectionError as RedisConnectionError

from database import SessionLocal, CertificateDB, BulkUploadBatchDB, VectorDB

logger = logging.getLogger(__name__)

# Redis configuration
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
REDIS_QUEUE_KEY = 'certificate_processing_queue'
REDIS_PROCESSING_KEY = 'certificate_processing_active'
REDIS_FAILED_KEY = 'certificate_processing_failed'
REDIS_RETRY_KEY = 'certificate_processing_retry'

# Configuration - Optimized for API stability
MAX_WORKERS = int(os.getenv('CERTIFICATE_WORKERS', '3'))  # Parallel workers (reduced to avoid rate limits)
WORKER_TIMEOUT = 600  # 10 minutes timeout per certificate
RATE_LIMIT_DELAY = float(os.getenv('CERTIFICATE_PROCESSING_DELAY', '0.5'))  # Delay between certificates (reduced from 10s)
MAX_RETRIES = int(os.getenv('CERTIFICATE_MAX_RETRIES', '3'))  # Max retries per certificate
RETRY_DELAY_BASE = int(os.getenv('CERTIFICATE_RETRY_DELAY', '30'))  # Base delay between retries in seconds

logger.info(f"📊 Certificate Queue Configuration:")
logger.info(f"  • Max Workers: {MAX_WORKERS}")
logger.info(f"  • Rate Limit Delay: {RATE_LIMIT_DELAY}s between certificates")
logger.info(f"  • Max Retries: {MAX_RETRIES}")
logger.info(f"  • Worker Timeout: {WORKER_TIMEOUT}s")

# Track active workers
active_workers: Dict[str, threading.Thread] = {}
worker_shutdown = threading.Event()

# Redis connection pool
_redis_pool = None


def get_redis_connection() -> redis.Redis:
    """Get a Redis connection from the connection pool."""
    global _redis_pool

    if _redis_pool is None:
        _redis_pool = redis.ConnectionPool.from_url(
            REDIS_URL,
            max_connections=MAX_WORKERS + 5,
            decode_responses=True
        )

    return redis.Redis(connection_pool=_redis_pool)


def is_redis_available() -> bool:
    """Check if Redis is available."""
    try:
        r = get_redis_connection()
        r.ping()
        return True
    except (RedisConnectionError, Exception) as e:
        logger.warning(f"Redis not available: {e}")
        return False


class CertificateTask:
    """Represents a certificate processing task."""

    def __init__(
        self,
        task_id: str,
        user_id: str,
        file_path: str,
        filename: str,
        batch_id: Optional[str] = None,
        file_hash: Optional[str] = None,
        file_size: Optional[int] = None,
        s3_key: Optional[str] = None,
        s3_url: Optional[str] = None,
        retry_count: int = 0,
        created_at: Optional[str] = None,
        last_error: Optional[str] = None
    ):
        self.task_id = task_id
        self.user_id = user_id
        self.file_path = file_path
        self.filename = filename
        self.batch_id = batch_id
        self.file_hash = file_hash
        self.file_size = file_size
        self.s3_key = s3_key
        self.s3_url = s3_url
        self.retry_count = retry_count
        self.created_at = created_at or datetime.utcnow().isoformat()
        self.last_error = last_error

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary format."""
        return {
            'task_id': self.task_id,
            'user_id': self.user_id,
            'file_path': self.file_path,
            'filename': self.filename,
            'batch_id': self.batch_id,
            'file_hash': self.file_hash,
            'file_size': self.file_size,
            's3_key': self.s3_key,
            's3_url': self.s3_url,
            'retry_count': self.retry_count,
            'created_at': self.created_at,
            'last_error': self.last_error
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CertificateTask':
        """Create a CertificateTask from a dictionary."""
        return cls(
            task_id=data['task_id'],
            user_id=data['user_id'],
            file_path=data['file_path'],
            filename=data['filename'],
            batch_id=data.get('batch_id'),
            file_hash=data.get('file_hash'),
            file_size=data.get('file_size'),
            s3_key=data.get('s3_key'),
            s3_url=data.get('s3_url'),
            retry_count=data.get('retry_count', 0),
            created_at=data.get('created_at'),
            last_error=data.get('last_error')
        )


def process_certificate_task(task: CertificateTask) -> bool:
    """
    Process a single certificate task.

    Args:
        task: CertificateTask instance to process

    Returns:
        True if successful, False if failed
    """
    # Import here to avoid circular imports
    from certificate_processor import certificate_processor

    db = SessionLocal()
    certificate_id = str(uuid.uuid4())
    success = False

    try:
        logger.info(f"Processing certificate task {task.task_id}: {task.filename} (attempt {task.retry_count + 1}/{MAX_RETRIES + 1})")

        # Update batch status to in_progress
        if task.batch_id:
            try:
                batch = db.query(BulkUploadBatchDB).filter(
                    BulkUploadBatchDB.id == task.batch_id
                ).with_for_update().first()  # Row-level locking

                if batch:
                    batch.in_progress_count += 1
                    if batch.status == 'queued':
                        batch.status = 'processing'
                        batch.started_at = datetime.utcnow()
                    db.commit()
            except Exception as batch_error:
                logger.error(f"Failed to update batch status: {batch_error}")
                db.rollback()
                # Continue processing even if batch update fails

        # Create initial certificate record with processing status
        certificate = CertificateDB(
            id=certificate_id,
            user_id=task.user_id,
            batch_id=task.batch_id,
            project_name=f"Processing {task.filename}",
            original_filename=task.filename,
            file_path=task.file_path,
            s3_key=task.s3_key,
            s3_url=task.s3_url,
            processing_status="processing",
            created_at=datetime.utcnow()
        )
        db.add(certificate)
        db.commit()

        # Process the certificate (this is the long-running operation)
        processed_id = certificate_processor.process_certificate(
            user_id=task.user_id,
            file_path=task.file_path,
            filename=task.filename,
            file_hash=task.file_hash,
            file_size=task.file_size,
            s3_key=task.s3_key,
            s3_url=task.s3_url
        )

        # Update the batch on success
        if task.batch_id:
            try:
                batch = db.query(BulkUploadBatchDB).filter(
                    BulkUploadBatchDB.id == task.batch_id
                ).with_for_update().first()

                if batch:
                    batch.in_progress_count = max(0, batch.in_progress_count - 1)
                    batch.processed_count += 1
                    batch.success_count += 1

                    # Check if batch is complete
                    if batch.processed_count >= batch.total_files:
                        batch.status = 'completed'
                        batch.completed_at = datetime.utcnow()

                    db.commit()
            except Exception as batch_error:
                logger.error(f"Failed to update batch status on success: {batch_error}")
                db.rollback()

        # Update certificate with batch_id if needed
        if task.batch_id:
            processed_cert = db.query(CertificateDB).filter(
                CertificateDB.id == processed_id
            ).first()
            if processed_cert:
                processed_cert.batch_id = task.batch_id
                db.commit()

        # Delete the initial placeholder certificate
        db.query(CertificateDB).filter(CertificateDB.id == certificate_id).delete()
        db.commit()

        logger.info(f"✅ Successfully processed certificate {processed_id} from task {task.task_id}")
        success = True

    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Failed to process certificate task {task.task_id}: {error_msg}")

        # Store error for retry tracking
        task.last_error = error_msg

        # Update batch status
        if task.batch_id:
            try:
                batch = db.query(BulkUploadBatchDB).filter(
                    BulkUploadBatchDB.id == task.batch_id
                ).with_for_update().first()

                if batch:
                    batch.in_progress_count = max(0, batch.in_progress_count - 1)

                    # Only mark as failed if we've exhausted all retries
                    if task.retry_count >= MAX_RETRIES:
                        batch.processed_count += 1
                        batch.failed_count += 1

                        # Add error to batch errors list
                        errors = batch.errors or []
                        errors.append({
                            'filename': task.filename,
                            'error': error_msg,
                            'timestamp': datetime.utcnow().isoformat(),
                            'attempts': task.retry_count + 1
                        })
                        batch.errors = errors

                        # Check if batch is complete (even with failures)
                        if batch.processed_count >= batch.total_files:
                            if batch.failed_count == batch.total_files:
                                batch.status = 'failed'
                            else:
                                batch.status = 'completed'
                            batch.completed_at = datetime.utcnow()

                    db.commit()
            except Exception as batch_error:
                logger.error(f"Failed to update batch status: {batch_error}")

        # Delete the placeholder certificate on failure
        try:
            placeholder_cert = db.query(CertificateDB).filter(
                CertificateDB.id == certificate_id
            ).first()
            if placeholder_cert:
                db.query(VectorDB).filter(VectorDB.certificate_id == certificate_id).delete()
                db.delete(placeholder_cert)
                db.commit()
                logger.info(f"Deleted placeholder certificate {certificate_id} after failure")
        except Exception as delete_error:
            logger.error(f"Failed to delete placeholder certificate: {delete_error}")
            db.rollback()

        success = False

    finally:
        db.close()

    return success


def certificate_worker(worker_id: int) -> None:
    """
    Background worker that processes certificates from Redis queue.

    Args:
        worker_id: Unique identifier for this worker thread
    """
    logger.info(f"🚀 Certificate worker {worker_id} started (Redis-backed queue)")

    while not worker_shutdown.is_set():
        try:
            r = get_redis_connection()

            # Try to get a task from the queue (blocking with timeout)
            result = r.blpop(REDIS_QUEUE_KEY, timeout=1)

            if result is None:
                # No tasks available, check for retry tasks
                retry_task = r.lpop(REDIS_RETRY_KEY)
                if retry_task:
                    try:
                        task_data = json.loads(retry_task)
                        task = CertificateTask.from_dict(task_data)

                        # Check if retry delay has passed
                        retry_delay = RETRY_DELAY_BASE * (2 ** task.retry_count)  # Exponential backoff
                        created_time = datetime.fromisoformat(task.created_at)
                        elapsed = (datetime.utcnow() - created_time).total_seconds()

                        if elapsed < retry_delay:
                            # Not ready for retry yet, put it back
                            r.rpush(REDIS_RETRY_KEY, retry_task)
                            continue

                        logger.info(f"Worker {worker_id} retrying task {task.task_id} (attempt {task.retry_count + 1})")

                        # Mark task as being processed
                        r.hset(REDIS_PROCESSING_KEY, task.task_id, json.dumps(task.to_dict()))

                        # Process the task
                        success = process_certificate_task(task)

                        # Remove from processing set
                        r.hdel(REDIS_PROCESSING_KEY, task.task_id)

                        if not success:
                            handle_failed_task(task, r)

                        # Rate limit delay
                        if RATE_LIMIT_DELAY > 0:
                            time.sleep(RATE_LIMIT_DELAY)
                    except json.JSONDecodeError:
                        logger.error(f"Worker {worker_id}: Invalid retry task JSON")
                        continue
                continue

            _, task_json = result

            try:
                task_data = json.loads(task_json)
                task = CertificateTask.from_dict(task_data)
            except json.JSONDecodeError:
                logger.error(f"Worker {worker_id}: Invalid task JSON in queue")
                continue

            logger.info(f"Worker {worker_id} picked up task {task.task_id}: {task.filename}")

            # Mark task as being processed
            r.hset(REDIS_PROCESSING_KEY, task.task_id, json.dumps(task.to_dict()))

            # Process the task
            success = process_certificate_task(task)

            # Remove from processing set
            r.hdel(REDIS_PROCESSING_KEY, task.task_id)

            if not success:
                handle_failed_task(task, r)

            # Add delay to avoid OpenAI API rate limiting
            if RATE_LIMIT_DELAY > 0:
                logger.debug(f"Worker {worker_id} waiting {RATE_LIMIT_DELAY} seconds to avoid rate limiting...")
                time.sleep(RATE_LIMIT_DELAY)

        except RedisConnectionError as e:
            logger.error(f"Worker {worker_id} Redis connection error: {e}")
            time.sleep(5)  # Wait before retrying
            continue
        except Exception as e:
            logger.error(f"Worker {worker_id} encountered error: {e}")
            time.sleep(1)
            continue

    logger.info(f"🛑 Certificate worker {worker_id} stopped")


def handle_failed_task(task: CertificateTask, r: redis.Redis) -> None:
    """
    Handle a failed certificate task - retry or move to failed queue.

    Args:
        task: The failed CertificateTask
        r: Redis connection
    """
    task.retry_count += 1

    if task.retry_count <= MAX_RETRIES:
        # Schedule for retry with exponential backoff
        task.created_at = datetime.utcnow().isoformat()  # Reset for retry timing
        logger.warning(f"⏳ Task {task.task_id} will retry (attempt {task.retry_count}/{MAX_RETRIES}): {task.filename}")
        r.rpush(REDIS_RETRY_KEY, json.dumps(task.to_dict()))
    else:
        # Move to failed queue for manual intervention
        logger.error(f"💀 Task {task.task_id} permanently failed after {MAX_RETRIES} retries: {task.filename}")
        r.hset(REDIS_FAILED_KEY, task.task_id, json.dumps(task.to_dict()))

        # Clean up file only after exhausting all retries
        try:
            if os.path.exists(task.file_path):
                # Don't delete - keep for manual retry
                logger.info(f"Keeping failed certificate file for manual retry: {task.file_path}")
        except Exception as cleanup_error:
            logger.error(f"Error handling failed file {task.file_path}: {cleanup_error}")


def start_workers(num_workers: int = MAX_WORKERS) -> None:
    """
    Start background worker threads for certificate processing.

    Args:
        num_workers: Number of worker threads to start
    """
    global active_workers

    if not is_redis_available():
        logger.error("❌ Redis is not available. Certificate processing will not work.")
        logger.error("   Please start Redis: brew services start redis (Mac) or redis-server (Linux)")
        return

    # Clear shutdown flag
    worker_shutdown.clear()

    # Recover any tasks that were being processed when server crashed
    recover_incomplete_tasks()

    # Start worker threads
    for i in range(num_workers):
        worker_id = f"worker_{i+1}"
        if worker_id not in active_workers or not active_workers[worker_id].is_alive():
            thread = threading.Thread(
                target=certificate_worker,
                args=(i+1,),
                name=f"CertificateWorker-{i+1}",
                daemon=True
            )
            thread.start()
            active_workers[worker_id] = thread
            logger.info(f"Started certificate worker {worker_id}")

    logger.info(f"✅ Certificate processing system started with {num_workers} workers (Redis-backed)")


def stop_workers() -> None:
    """Stop all background worker threads gracefully."""
    global active_workers

    logger.info("Stopping certificate workers...")

    # Set shutdown flag
    worker_shutdown.set()

    # Wait for workers to finish
    for worker_id, thread in active_workers.items():
        thread.join(timeout=5)
        if thread.is_alive():
            logger.warning(f"Worker {worker_id} did not stop gracefully")

    active_workers.clear()
    logger.info("All certificate workers stopped")


def recover_incomplete_tasks() -> None:
    """
    Recover tasks that were being processed when the server crashed.
    Move them back to the main queue for reprocessing.
    """
    try:
        r = get_redis_connection()

        # Get all tasks that were being processed
        processing_tasks = r.hgetall(REDIS_PROCESSING_KEY)

        if processing_tasks:
            logger.info(f"🔄 Recovering {len(processing_tasks)} incomplete tasks from previous session")

            for task_id, task_json in processing_tasks.items():
                try:
                    task_data = json.loads(task_json)
                    task = CertificateTask.from_dict(task_data)

                    # Re-queue the task
                    r.rpush(REDIS_QUEUE_KEY, json.dumps(task.to_dict()))
                    r.hdel(REDIS_PROCESSING_KEY, task_id)

                    logger.info(f"Recovered task {task_id}: {task.filename}")
                except Exception as e:
                    logger.error(f"Failed to recover task {task_id}: {e}")
                    r.hdel(REDIS_PROCESSING_KEY, task_id)

    except Exception as e:
        logger.error(f"Error during task recovery: {e}")


def enqueue_certificate(
    user_id: str,
    file_path: str,
    filename: str,
    batch_id: Optional[str] = None,
    file_hash: Optional[str] = None,
    file_size: Optional[int] = None,
    s3_key: Optional[str] = None,
    s3_url: Optional[str] = None
) -> str:
    """
    Add a certificate to the Redis processing queue.

    Args:
        user_id: ID of the user uploading the certificate
        file_path: Path to the saved certificate file (local, may not exist)
        filename: Original filename
        batch_id: Optional batch ID for tracking bulk uploads
        file_hash: Optional SHA256 hash for duplicate detection
        file_size: Optional file size in bytes
        s3_key: Optional S3 object key for the file
        s3_url: Optional S3 URL for the file

    Returns:
        Task ID for tracking
    """
    task_id = str(uuid.uuid4())
    task = CertificateTask(
        task_id=task_id,
        user_id=user_id,
        file_path=file_path,
        filename=filename,
        batch_id=batch_id,
        file_hash=file_hash,
        file_size=file_size,
        s3_key=s3_key,
        s3_url=s3_url
    )

    try:
        r = get_redis_connection()
        r.rpush(REDIS_QUEUE_KEY, json.dumps(task.to_dict()))
        logger.info(f"📥 Enqueued certificate task {task_id}: {filename}")
    except RedisConnectionError as e:
        logger.error(f"Failed to enqueue task (Redis unavailable): {e}")
        raise RuntimeError("Certificate processing queue unavailable. Please try again later.")

    return task_id


def get_queue_status() -> Dict[str, Any]:
    """
    Get current status of the certificate processing queue.

    Returns:
        Dictionary with queue statistics
    """
    try:
        r = get_redis_connection()

        return {
            'queue_size': r.llen(REDIS_QUEUE_KEY),
            'processing_count': r.hlen(REDIS_PROCESSING_KEY),
            'retry_queue_size': r.llen(REDIS_RETRY_KEY),
            'failed_count': r.hlen(REDIS_FAILED_KEY),
            'active_workers': len([w for w in active_workers.values() if w.is_alive()]),
            'total_workers': len(active_workers),
            'is_running': not worker_shutdown.is_set(),
            'redis_connected': True
        }
    except RedisConnectionError:
        return {
            'queue_size': 0,
            'processing_count': 0,
            'retry_queue_size': 0,
            'failed_count': 0,
            'active_workers': 0,
            'total_workers': len(active_workers),
            'is_running': False,
            'redis_connected': False
        }


def get_failed_tasks() -> List[Dict[str, Any]]:
    """
    Get all permanently failed tasks for manual review.

    Returns:
        List of failed task dictionaries
    """
    try:
        r = get_redis_connection()
        failed_tasks = r.hgetall(REDIS_FAILED_KEY)

        return [
            json.loads(task_json)
            for task_json in failed_tasks.values()
        ]
    except Exception as e:
        logger.error(f"Error getting failed tasks: {e}")
        return []


def retry_failed_task(task_id: str) -> bool:
    """
    Manually retry a failed task.

    Args:
        task_id: ID of the failed task to retry

    Returns:
        True if task was re-queued, False otherwise
    """
    try:
        r = get_redis_connection()

        task_json = r.hget(REDIS_FAILED_KEY, task_id)
        if not task_json:
            logger.warning(f"Failed task {task_id} not found")
            return False

        task_data = json.loads(task_json)
        task_data['retry_count'] = 0  # Reset retry count
        task_data['last_error'] = None
        task_data['created_at'] = datetime.utcnow().isoformat()

        # Move from failed to main queue
        r.hdel(REDIS_FAILED_KEY, task_id)
        r.rpush(REDIS_QUEUE_KEY, json.dumps(task_data))

        logger.info(f"🔄 Re-queued failed task {task_id}")
        return True

    except Exception as e:
        logger.error(f"Error retrying failed task {task_id}: {e}")
        return False


def clear_failed_tasks() -> int:
    """
    Clear all permanently failed tasks.

    Returns:
        Number of tasks cleared
    """
    try:
        r = get_redis_connection()
        count = r.hlen(REDIS_FAILED_KEY)
        r.delete(REDIS_FAILED_KEY)
        logger.info(f"Cleared {count} failed tasks")
        return count
    except Exception as e:
        logger.error(f"Error clearing failed tasks: {e}")
        return 0


# Auto-start workers when module is imported
start_workers()
