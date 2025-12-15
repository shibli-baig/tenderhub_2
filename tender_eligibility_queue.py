"""
Background tender eligibility processing queue system.

This module provides a thread-based worker queue for processing tender documents
and extracting eligibility criteria asynchronously.
"""

import os
import logging
import threading
import queue
import uuid
import time
from typing import Dict, Any, Optional
from datetime import datetime

from tender_eligibility_processor import tender_eligibility_processor

logger = logging.getLogger(__name__)

# Global queue for tender analysis tasks
tender_analysis_queue: queue.Queue = queue.Queue()

# Track active workers
active_workers: Dict[str, threading.Thread] = {}
worker_shutdown = threading.Event()

# Configuration
MAX_WORKERS = int(os.getenv('TENDER_ANALYSIS_WORKERS', '2'))  # Fewer workers due to heavy processing
RATE_LIMIT_DELAY = int(os.getenv('TENDER_ANALYSIS_DELAY', '10'))  # 10 seconds between analyses


class TenderAnalysisTask:
    """Represents a tender analysis task."""

    def __init__(
        self,
        task_id: str,
        tender_id: str,
        analysis_status_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        self.task_id = task_id
        self.tender_id = tender_id
        self.analysis_status_id = analysis_status_id
        self.user_id = user_id
        self.created_at = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary format."""
        return {
            'task_id': self.task_id,
            'tender_id': self.tender_id,
            'analysis_status_id': self.analysis_status_id,
            'user_id': self.user_id,
            'created_at': self.created_at.isoformat()
        }


def process_tender_analysis_task(task: TenderAnalysisTask) -> None:
    """
    Process a single tender analysis task.

    Args:
        task: TenderAnalysisTask instance to process
    """
    try:
        logger.info(f"Processing tender analysis task {task.task_id}: tender {task.tender_id}")

        # Process the tender eligibility
        analysis_status_id = tender_eligibility_processor.process_tender_eligibility(
            tender_id=task.tender_id,
            analysis_status_id=task.analysis_status_id,
        )

        logger.info(f"Successfully completed tender analysis {analysis_status_id} for task {task.task_id}")

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Failed to process tender analysis task {task.task_id}: {error_msg}")


def tender_analysis_worker(worker_id: int) -> None:
    """
    Background worker that processes tender analysis tasks from the queue.

    Args:
        worker_id: Unique identifier for this worker thread
    """
    logger.info(f"Tender analysis worker {worker_id} started")

    while not worker_shutdown.is_set():
        try:
            # Get task from queue with timeout
            task = tender_analysis_queue.get(timeout=1)

            if task is None:  # Shutdown signal
                logger.info(f"Worker {worker_id} received shutdown signal")
                break

            logger.info(f"Worker {worker_id} picked up task {task.task_id}")

            # Process the task
            process_tender_analysis_task(task)

            # Add delay to avoid API rate limiting
            if RATE_LIMIT_DELAY > 0:
                logger.info(f"Worker {worker_id} waiting {RATE_LIMIT_DELAY} seconds to avoid rate limiting...")
                time.sleep(RATE_LIMIT_DELAY)

            # Mark task as done
            tender_analysis_queue.task_done()

        except queue.Empty:
            # No tasks available, continue waiting
            continue
        except Exception as e:
            logger.error(f"Worker {worker_id} encountered error: {e}")
            # Don't crash the worker, continue processing
            continue

    logger.info(f"Tender analysis worker {worker_id} stopped")


def start_workers(num_workers: int = MAX_WORKERS) -> None:
    """
    Start background worker threads for tender analysis processing.

    Args:
        num_workers: Number of worker threads to start
    """
    global active_workers

    # Clear shutdown flag
    worker_shutdown.clear()

    # Start worker threads
    for i in range(num_workers):
        worker_id = f"worker_{i+1}"
        if worker_id not in active_workers or not active_workers[worker_id].is_alive():
            thread = threading.Thread(
                target=tender_analysis_worker,
                args=(i+1,),
                name=f"TenderAnalysisWorker-{i+1}",
                daemon=True
            )
            thread.start()
            active_workers[worker_id] = thread
            logger.info(f"Started tender analysis worker {worker_id}")

    logger.info(f"Tender analysis system started with {num_workers} workers")


def stop_workers() -> None:
    """Stop all background worker threads gracefully."""
    global active_workers

    logger.info("Stopping tender analysis workers...")

    # Set shutdown flag
    worker_shutdown.set()

    # Send shutdown signals to all workers
    for _ in range(len(active_workers)):
        tender_analysis_queue.put(None)

    # Wait for workers to finish
    for worker_id, thread in active_workers.items():
        thread.join(timeout=5)
        if thread.is_alive():
            logger.warning(f"Worker {worker_id} did not stop gracefully")

    active_workers.clear()
    logger.info("All tender analysis workers stopped")


def enqueue_tender_analysis(
    tender_id: str,
    user_id: Optional[str] = None,
    analysis_status_id: Optional[str] = None,
) -> str:
    """
    Add a tender to the analysis queue.

    Args:
        tender_id: ID of the tender to analyze
        user_id: Optional ID of the user requesting analysis

    Returns:
        Task ID for tracking
    """
    task_id = str(uuid.uuid4())
    task = TenderAnalysisTask(
        task_id=task_id,
        tender_id=tender_id,
        analysis_status_id=analysis_status_id,
        user_id=user_id
    )

    tender_analysis_queue.put(task)
    logger.info(f"Enqueued tender analysis task {task_id}: tender {tender_id}")

    return task_id


def get_queue_status() -> Dict[str, Any]:
    """
    Get current status of the tender analysis queue.

    Returns:
        Dictionary with queue statistics
    """
    return {
        'queue_size': tender_analysis_queue.qsize(),
        'active_workers': len([w for w in active_workers.values() if w.is_alive()]),
        'total_workers': len(active_workers),
        'is_running': not worker_shutdown.is_set()
    }


# Auto-start workers when module is imported
start_workers()
