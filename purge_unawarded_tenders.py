"""
Production-ready utility script to purge all non-awarded tenders from the database and S3.

This script:
- Connects to Render PostgreSQL database (via DATABASE_URL env var)
- Connects to AWS S3 (via AWS credentials from env vars)
- Deletes all non-awarded tenders and their related records from PostgreSQL
- Deletes all associated S3 files (screenshots, documents) for purged tenders
- Provides detailed logging and error handling for production use

Usage:
    # Preview what would be deleted (dry run)
    python purge_unawarded_tenders.py --dry-run
    
    # Actually purge (requires confirmation)
    python purge_unawarded_tenders.py --confirm

Only tenders flagged as awarded are retained. The script removes:
- Associated favorites, shortlisted entries, stage documents
- Task comments, tasks, tender messages
- Notifications, certificate attachments
- All S3 files associated with the tenders
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict, List, Tuple, Optional, Any
from sqlalchemy.orm import Session
from sqlalchemy import text
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Import database models and session
from database import (
    SessionLocal,
    TenderDB,
    ShortlistedTenderDB,
    StageDocumentDB,
    TenderAssignmentDB,
    TaskDB,
    TaskCommentDB,
    TenderMessageDB,
    NotificationDB,
    TenderCertificateAttachmentDB,
    engine,
)

# Import S3 storage manager
from core.s3_storage import s3_manager


def list_tender_s3_files(tender_id: str) -> List[str]:
    """
    List all S3 files associated with a tender.
    
    Args:
        tender_id: The tender ID to list files for
        
    Returns:
        List of S3 keys (file paths) for the tender
        
    Raises:
        Exception: If S3 is not available or listing fails
    """
    if not s3_manager.enabled:
        raise RuntimeError(f"S3 storage not enabled - cannot list files for tender {tender_id}")
    
    s3_client = s3_manager.s3_client
    if not s3_client:
        raise RuntimeError(f"S3 client not available - cannot list files for tender {tender_id}")
    
    s3_files = []
    prefix = f"tenders/{tender_id}/"
    
    try:
        # List all objects with the tender prefix
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(
            Bucket=s3_manager.bucket_name,
            Prefix=prefix
        )
        
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    s3_files.append(obj['Key'])
        
        logger.debug(f"Found {len(s3_files)} S3 files for tender {tender_id}")
        return s3_files
        
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        error_msg = f"Failed to list S3 files for tender {tender_id}: {error_code} - {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e
    except Exception as e:
        error_msg = f"Unexpected error listing S3 files for tender {tender_id}: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def delete_tender_s3_files(tender_id: str, dry_run: bool = False) -> Tuple[int, int]:
    """
    Delete all S3 files associated with a tender.
    
    Args:
        tender_id: The tender ID whose files should be deleted
        dry_run: If True, only log what would be deleted without actually deleting
        
    Returns:
        Tuple of (success_count, failed_count)
        
    Raises:
        RuntimeError: If S3 is not available or deletion fails critically
    """
    if not s3_manager.enabled:
        raise RuntimeError(f"S3 storage not enabled - cannot delete files for tender {tender_id}")
    
    if not s3_manager.s3_client:
        raise RuntimeError(f"S3 client not available - cannot delete files for tender {tender_id}")
    
    s3_files = list_tender_s3_files(tender_id)
    
    if not s3_files:
        logger.debug(f"No S3 files found for tender {tender_id}")
        return (0, 0)
    
    if dry_run:
        logger.info(f"  [DRY RUN] Would delete {len(s3_files)} S3 files for tender {tender_id}")
        for s3_key in s3_files:
            logger.debug(f"    [DRY RUN] Would delete: {s3_key}")
        return (len(s3_files), 0)
    
    success_count = 0
    failed_count = 0
    failed_keys = []
    
    for s3_key in s3_files:
        try:
            if s3_manager.delete_file(s3_key):
                success_count += 1
                logger.debug(f"    Deleted S3 file: {s3_key}")
            else:
                failed_count += 1
                failed_keys.append(s3_key)
                logger.error(f"    FAILED to delete S3 file: {s3_key}")
        except Exception as e:
            failed_count += 1
            failed_keys.append(s3_key)
            logger.error(f"    ERROR deleting S3 file {s3_key}: {e}")
    
    if failed_count > 0:
        error_msg = f"Failed to delete {failed_count}/{len(s3_files)} S3 files for tender {tender_id}"
        logger.error(f"  {error_msg}")
        logger.error(f"  Failed files: {failed_keys[:5]}{'...' if len(failed_keys) > 5 else ''}")
        # Don't raise exception here - log and continue, but track failures
        # The script will report total failures at the end
    
    if success_count > 0:
        logger.info(f"  Successfully deleted {success_count}/{len(s3_files)} S3 files for tender {tender_id}")
    
    return (success_count, failed_count)


def purge_non_awarded_tenders(db: Session, dry_run: bool = False) -> Dict[str, Any]:
    """
    Delete every tender that is not awarded, along with S3 files and related records.
    
    Args:
        db: Database session
        dry_run: If True, only log what would be deleted without actually deleting
        
    Returns:
        Dictionary with purge statistics:
        {
            'tenders_purged': int,
            's3_files_deleted': int,
            's3_files_failed': int,
            'db_records_deleted': int,
            'errors': List[str]
        }
    """
    stats = {
        'tenders_purged': 0,
        's3_files_deleted': 0,
        's3_files_failed': 0,
        'db_records_deleted': 0,
        'errors': []
    }
    
    # Find all non-awarded tenders
    tenders = db.query(TenderDB).filter(TenderDB.awarded != True).all()  # noqa: E712
    
    if not tenders:
        logger.info("No non-awarded tenders found to purge.")
        return stats
    
    tender_count = len(tenders)
    logger.info(f"Found {tender_count} non-awarded tender(s) to purge.")
    
    if dry_run:
        logger.info("[DRY RUN MODE] - No actual deletions will be performed")
    
    tender_ids = [t.id for t in tenders]
    
    # Delete S3 files for each tender - MANDATORY
    logger.info("Deleting S3 files for non-awarded tenders...")
    for idx, tender in enumerate(tenders, 1):
        logger.info(f"Processing tender {tender.id}... ({idx}/{tender_count})")
        
        try:
            success, failed = delete_tender_s3_files(tender.id, dry_run=dry_run)
            stats['s3_files_deleted'] += success
            stats['s3_files_failed'] += failed
            
            # If we have failures and this is not a dry run, log as error
            if failed > 0 and not dry_run:
                error_msg = f"Failed to delete {failed} S3 file(s) for tender {tender.id}"
                logger.error(error_msg)
                stats['errors'].append(error_msg)
        except RuntimeError as e:
            # RuntimeError means S3 is not available - this is CRITICAL
            error_msg = f"CRITICAL: S3 deletion failed for tender {tender.id}: {e}"
            logger.error(error_msg)
            stats['errors'].append(error_msg)
            # Don't continue if S3 is completely unavailable
            if "not enabled" in str(e) or "not available" in str(e):
                raise RuntimeError(f"Cannot proceed: S3 is required but not available. {e}")
        except Exception as e:
            error_msg = f"Error deleting S3 files for tender {tender.id}: {e}"
            logger.error(error_msg)
            stats['errors'].append(error_msg)
            # For other exceptions, continue but track the error
    
    if not dry_run:
        logger.info(f"Completed S3 file deletion: {stats['s3_files_deleted']} deleted, {stats['s3_files_failed']} failed")
    
    # Now delete database records
    logger.info("Deleting database records...")
    
    try:
        # Remove stage documents before shortlists get deleted (FK constraint).
        shortlist_ids = [
            row[0]
            for row in db.query(ShortlistedTenderDB.id)
            .filter(ShortlistedTenderDB.tender_id.in_(tender_ids))
            .all()
        ]
        if shortlist_ids:
            if dry_run:
                count = db.query(StageDocumentDB).filter(StageDocumentDB.shortlist_id.in_(shortlist_ids)).count()
                logger.info(f"  [DRY RUN] Would delete {count} stage documents")
            else:
                deleted = db.query(StageDocumentDB).filter(StageDocumentDB.shortlist_id.in_(shortlist_ids)).delete(
                    synchronize_session=False
                )
                stats['db_records_deleted'] += deleted
                logger.info(f"  Deleted {deleted} stage documents")
        
        # Assignment/task graph cleanup
        assignment_ids = [
            row[0]
            for row in db.query(TenderAssignmentDB.id)
            .filter(TenderAssignmentDB.tender_id.in_(tender_ids))
            .all()
        ]
        
        if assignment_ids:
            task_ids = [
                row[0]
                for row in db.query(TaskDB.id)
                .filter(TaskDB.assignment_id.in_(assignment_ids))
                .all()
            ]
            
            if task_ids:
                if dry_run:
                    comment_count = db.query(TaskCommentDB).filter(TaskCommentDB.task_id.in_(task_ids)).count()
                    task_count = len(task_ids)
                    logger.info(f"  [DRY RUN] Would delete {comment_count} task comments and {task_count} tasks")
                else:
                    deleted_comments = db.query(TaskCommentDB).filter(TaskCommentDB.task_id.in_(task_ids)).delete(
                        synchronize_session=False
                    )
                    deleted_tasks = db.query(TaskDB).filter(TaskDB.id.in_(task_ids)).delete(synchronize_session=False)
                    stats['db_records_deleted'] += deleted_comments + deleted_tasks
                    logger.info(f"  Deleted {deleted_comments} task comments and {deleted_tasks} tasks")
            
            if dry_run:
                msg_count = db.query(TenderMessageDB).filter(TenderMessageDB.assignment_id.in_(assignment_ids)).count()
                assignment_count = len(assignment_ids)
                logger.info(f"  [DRY RUN] Would delete {msg_count} tender messages and {assignment_count} assignments")
            else:
                deleted_messages = db.query(TenderMessageDB).filter(TenderMessageDB.assignment_id.in_(assignment_ids)).delete(
                    synchronize_session=False
                )
                deleted_assignments = db.query(TenderAssignmentDB).filter(TenderAssignmentDB.id.in_(assignment_ids)).delete(
                    synchronize_session=False
                )
                stats['db_records_deleted'] += deleted_messages + deleted_assignments
                logger.info(f"  Deleted {deleted_messages} tender messages and {deleted_assignments} assignments")
        
        # Clean up reminder/notification dependencies.
        if dry_run:
            notif_count = db.query(NotificationDB).filter(NotificationDB.tender_id.in_(tender_ids)).count()
            logger.info(f"  [DRY RUN] Would delete {notif_count} notifications")
        else:
            deleted_notifs = db.query(NotificationDB).filter(NotificationDB.tender_id.in_(tender_ids)).delete(
                synchronize_session=False
            )
            stats['db_records_deleted'] += deleted_notifs
            logger.info(f"  Deleted {deleted_notifs} notifications")
        
        # Clean up certificate attachments.
        if dry_run:
            cert_count = db.query(TenderCertificateAttachmentDB).filter(
                TenderCertificateAttachmentDB.tender_id.in_(tender_ids)
            ).count()
            logger.info(f"  [DRY RUN] Would delete {cert_count} certificate attachments")
        else:
            deleted_certs = db.query(TenderCertificateAttachmentDB).filter(
                TenderCertificateAttachmentDB.tender_id.in_(tender_ids)
            ).delete(synchronize_session=False)
            stats['db_records_deleted'] += deleted_certs
            logger.info(f"  Deleted {deleted_certs} certificate attachments")
        
        # Finally delete the tenders (cascades remove favourites, shortlist, etc.).
        if dry_run:
            logger.info(f"  [DRY RUN] Would delete {tender_count} tenders")
        else:
            for tender in tenders:
                db.delete(tender)
                stats['tenders_purged'] += 1
            
            db.commit()
            logger.info(f"  Deleted {tender_count} tenders from database")
        
    except Exception as e:
        if not dry_run:
            db.rollback()
        error_msg = f"Error during database deletion: {e}"
        logger.error(error_msg)
        stats['errors'].append(error_msg)
        raise
    
    return stats


def verify_connections() -> Tuple[bool, bool]:
    """
    Verify database and S3 connections are available.
    S3 connection is MANDATORY - script will fail if S3 is not available.
    
    Returns:
        Tuple of (db_connected, s3_connected)
    """
    db_connected = False
    s3_connected = False
    
    # Verify database connection
    try:
        if engine.url.get_backend_name() != "postgresql":
            logger.error("Database is not PostgreSQL. This script requires PostgreSQL.")
            return (False, False)
        
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_connected = True
        logger.info("✓ Database connection verified (PostgreSQL)")
    except Exception as e:
        logger.error(f"✗ Database connection failed: {e}")
        return (False, False)
    
    # Verify S3 connection - MANDATORY for this script
    if not s3_manager.enabled:
        logger.error("=" * 60)
        logger.error("S3 STORAGE NOT ENABLED - THIS IS REQUIRED!")
        logger.error("=" * 60)
        logger.error("S3 file deletion is a core function and MUST work.")
        logger.error("Please set USE_S3_STORAGE=True and configure AWS credentials.")
        logger.error("Required environment variables:")
        logger.error("  - USE_S3_STORAGE=True")
        logger.error("  - AWS_ACCESS_KEY_ID")
        logger.error("  - AWS_SECRET_ACCESS_KEY")
        logger.error("  - AWS_S3_BUCKET_NAME")
        logger.error("  - AWS_REGION")
        return (db_connected, False)
    
    if not s3_manager.s3_client:
        logger.error("=" * 60)
        logger.error("S3 CLIENT NOT INITIALIZED - THIS IS REQUIRED!")
        logger.error("=" * 60)
        logger.error("S3 file deletion is a core function and MUST work.")
        logger.error("Please check your AWS credentials and configuration.")
        return (db_connected, False)
    
    # Test S3 bucket access with multiple methods
    s3_client = s3_manager.s3_client
    bucket_name = s3_manager.bucket_name
    
    logger.info(f"Testing S3 connection to bucket: {bucket_name}")
    logger.info(f"S3 Region: {os.getenv('AWS_REGION', 'us-east-1')}")
    
    try:
        # Method 1: Try head_bucket first (most reliable)
        try:
            s3_client.head_bucket(Bucket=bucket_name)
            s3_connected = True
            logger.info(f"✓ S3 connection verified (bucket: {bucket_name})")
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            
            if error_code == '404':
                logger.error("=" * 60)
                logger.error("S3 BUCKET NOT FOUND - THIS IS REQUIRED!")
                logger.error("=" * 60)
                logger.error(f"Bucket '{bucket_name}' does not exist or is in a different region.")
                logger.error(f"Error: {error_message}")
                logger.error("")
                logger.error("Possible solutions:")
                logger.error("  1. Check if bucket name is correct (current: {})".format(bucket_name))
                logger.error("  2. Verify bucket exists in region: {}".format(os.getenv('AWS_REGION', 'us-east-1')))
                logger.error("  3. Try listing all buckets to verify access:")
                logger.error("     aws s3 ls")
                logger.error("  4. Check AWS credentials have s3:ListBucket and s3:DeleteObject permissions")
                
                # Try to list buckets to help diagnose
                try:
                    logger.info("Attempting to list available buckets...")
                    response = s3_client.list_buckets()
                    available_buckets = [b['Name'] for b in response.get('Buckets', [])]
                    logger.info(f"Available buckets: {', '.join(available_buckets) if available_buckets else 'None found'}")
                    if bucket_name not in available_buckets:
                        logger.error(f"Bucket '{bucket_name}' is NOT in the list of available buckets!")
                except Exception as list_error:
                    logger.error(f"Could not list buckets: {list_error}")
                
                return (db_connected, False)
                
            elif error_code == '403':
                logger.error("=" * 60)
                logger.error("S3 ACCESS DENIED - THIS IS REQUIRED!")
                logger.error("=" * 60)
                logger.error(f"Access denied to bucket '{bucket_name}'.")
                logger.error(f"Error: {error_message}")
                logger.error("")
                logger.error("Possible solutions:")
                logger.error("  1. Verify AWS credentials are correct")
                logger.error("  2. Check IAM policy has s3:ListBucket and s3:DeleteObject permissions")
                logger.error("  3. Verify bucket policy allows your AWS account access")
                return (db_connected, False)
            else:
                # Try alternative method: list_objects_v2
                logger.warning(f"head_bucket failed ({error_code}), trying list_objects_v2...")
                try:
                    s3_client.list_objects_v2(Bucket=bucket_name, MaxKeys=1)
                    s3_connected = True
                    logger.info(f"✓ S3 connection verified via list_objects_v2 (bucket: {bucket_name})")
                except ClientError as e2:
                    error_code2 = e2.response.get('Error', {}).get('Code', 'Unknown')
                    logger.error("=" * 60)
                    logger.error("S3 CONNECTION FAILED - THIS IS REQUIRED!")
                    logger.error("=" * 60)
                    logger.error(f"Both head_bucket and list_objects_v2 failed.")
                    logger.error(f"Error code: {error_code2}")
                    logger.error(f"Error: {e2}")
                    return (db_connected, False)
                    
    except Exception as e:
        logger.error("=" * 60)
        logger.error("S3 CONNECTION FAILED - THIS IS REQUIRED!")
        logger.error("=" * 60)
        logger.error(f"Unexpected error testing S3 connection: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return (db_connected, False)
    
    return (db_connected, s3_connected)


def main() -> None:
    """Entrypoint with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='Purge all non-awarded tenders from database and S3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview what would be deleted (safe, no changes)
  python purge_unawarded_tenders.py --dry-run
  
  # Actually purge (requires confirmation)
  python purge_unawarded_tenders.py --confirm
        """
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview what would be deleted without actually deleting anything'
    )
    parser.add_argument(
        '--confirm',
        action='store_true',
        help='Confirm that you want to proceed with the purge (required for actual deletion)'
    )
    
    args = parser.parse_args()
    
    # Verify connections
    logger.info("=" * 60)
    logger.info("PURGE NON-AWARDED TENDERS SCRIPT")
    logger.info("=" * 60)
    logger.info("Verifying connections...")
    
    db_connected, s3_connected = verify_connections()
    
    if not db_connected:
        logger.error("Cannot proceed: Database connection failed")
        sys.exit(1)
    
    if not s3_connected:
        logger.error("=" * 60)
        logger.error("CRITICAL ERROR: S3 CONNECTION FAILED")
        logger.error("=" * 60)
        logger.error("S3 file deletion is a CORE FUNCTION and MUST work.")
        logger.error("This script cannot proceed without S3 access.")
        logger.error("Please fix the S3 configuration and try again.")
        sys.exit(1)
    
    # Safety check: require confirmation for actual deletion
    if not args.dry_run and not args.confirm:
        logger.error("=" * 60)
        logger.error("SAFETY CHECK FAILED")
        logger.error("=" * 60)
        logger.error("This script will DELETE all non-awarded tenders and their S3 files.")
        logger.error("To proceed, you must use --confirm flag:")
        logger.error("  python purge_unawarded_tenders.py --confirm")
        logger.error("")
        logger.error("To preview what would be deleted (safe):")
        logger.error("  python purge_unawarded_tenders.py --dry-run")
        sys.exit(1)
    
    # Get database session
    db = SessionLocal()
    try:
        # Run purge
        stats = purge_non_awarded_tenders(db, dry_run=args.dry_run)
        
        # Print summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("PURGE SUMMARY")
        logger.info("=" * 60)
        
        if args.dry_run:
            logger.info("[DRY RUN MODE - No actual deletions performed]")
        
        logger.info(f"Tenders purged: {stats['tenders_purged']}")
        logger.info(f"S3 files deleted: {stats['s3_files_deleted']}")
        if stats['s3_files_failed'] > 0:
            logger.warning(f"S3 files failed: {stats['s3_files_failed']}")
        logger.info(f"Database records deleted: {stats['db_records_deleted']}")
        
        if stats['errors']:
            logger.warning(f"Errors encountered: {len(stats['errors'])}")
            for error in stats['errors']:
                logger.warning(f"  - {error}")
        
        logger.info("=" * 60)
        
        if args.dry_run:
            logger.info("Dry run completed. Use --confirm to actually perform the purge.")
        else:
            logger.info("Purge completed successfully!")
            
    except Exception as e:
        logger.error(f"Fatal error during purge: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()

