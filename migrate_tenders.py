#!/usr/bin/env python3
"""
Migration Script: NewScraper SQLite → Render PostgreSQL + S3
=============================================================
Migrates tenders and documents from the NewScraper SQLite database to
Render PostgreSQL and S3 storage.

Usage:
    python migrate_tenders.py              # Full migration
    python migrate_tenders.py --dry-run    # Preview without changes
    python migrate_tenders.py --skip-documents  # Tenders only
"""

import argparse
import json
import logging
import mimetypes
import os
import sqlite3
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database import Base, TenderDB, TenderDocumentDB
from s3_utils import upload_to_s3, delete_from_s3


def parse_migration_date(date_str: str) -> Optional[datetime]:
    """Parse date strings from SQLite, including ISO format."""
    if not date_str or str(date_str).upper() == 'NA':
        return None

    # Extended formats including ISO format
    formats = [
        "%Y-%m-%dT%H:%M:%S",      # ISO format: 2026-01-30T18:50:00
        "%Y-%m-%dT%H:%M:%S.%f",   # ISO with microseconds
        "%Y-%m-%d %H:%M:%S",      # Standard datetime
        "%d-%b-%Y %I:%M %p",      # 30-Jan-2026 06:50 PM
        "%d-%m-%Y %H:%M",         # 30-01-2026 18:50
        "%Y-%m-%d %H:%M",         # 2026-01-30 18:50
        "%d/%m/%Y",               # 30/01/2026
        "%Y-%m-%d",               # 2026-01-30
    ]

    date_str = str(date_str).strip()

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except (ValueError, AttributeError):
            continue

    # Try dateutil as fallback
    try:
        from dateutil import parser as date_parser
        return date_parser.parse(date_str)
    except Exception:
        pass

    logger.warning(f"Could not parse date: {date_str}")
    return None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('migration.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
SQLITE_PATH = "/Users/shiblibaig/Downloads/Python Codes/NewScraper/masterTenderDb.db"
DOCUMENT_BASE_PATH = "/Users/shiblibaig/Downloads/Python Codes/NewScraper/Master_Document_Directory"
PROGRESS_FILE = "migration_progress.json"
BATCH_SIZE = 10

# Portal URL to folder mapping
PORTAL_FOLDER_MAP = {
    "https://defproc.gov.in": "defproc_gov_in",
    "https://eproc.rajasthan.gov.in": "eproc_rajasthan_gov_in"
}

# MIME type mapping
MIME_TYPE_MAP = {
    ".pdf": "application/pdf",
    ".xls": "application/vnd.ms-excel",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".zip": "application/zip",
    ".rar": "application/x-rar-compressed",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".txt": "text/plain",
}


def get_mime_type(filename: str) -> str:
    """Get MIME type for a file."""
    ext = Path(filename).suffix.lower()
    if ext in MIME_TYPE_MAP:
        return MIME_TYPE_MAP[ext]
    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type or "application/octet-stream"


def get_document_type(filename: str) -> str:
    """Determine document type from filename."""
    ext = Path(filename).suffix.lower()
    type_map = {
        ".pdf": "pdf",
        ".xls": "excel",
        ".xlsx": "excel",
        ".doc": "word",
        ".docx": "word",
        ".zip": "archive",
        ".rar": "archive",
        ".png": "image",
        ".jpg": "image",
        ".jpeg": "image",
        ".gif": "image",
        ".txt": "text",
    }
    return type_map.get(ext, "other")


def parse_json_field(value: Any) -> Any:
    """Parse JSON field from SQLite."""
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    return value


def parse_float_value(value: Any) -> Optional[float]:
    """Parse float value from SQLite."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def load_progress() -> Dict[str, Any]:
    """Load migration progress from file."""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load progress file: {e}")
    return {"migrated_ids": [], "last_sqlite_id": 0, "started_at": None}


def save_progress(progress: Dict[str, Any]) -> None:
    """Save migration progress to file."""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def get_postgresql_session():
    """Create PostgreSQL session from DATABASE_URL."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")

    # Convert postgres:// to postgresql:// for SQLAlchemy 2.x
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)

    engine = create_engine(database_url, pool_pre_ping=True)
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal(), engine


def check_duplicate(session, tender_id: Optional[str], source_url: Optional[str]) -> bool:
    """Check if tender already exists in PostgreSQL."""
    if tender_id:
        existing = session.query(TenderDB).filter(TenderDB.tender_id == tender_id).first()
        if existing:
            return True

    if source_url:
        existing = session.query(TenderDB).filter(TenderDB.source_url == source_url).first()
        if existing:
            return True

    return False


def map_sqlite_to_postgresql(row: Dict[str, Any]) -> Dict[str, Any]:
    """Map SQLite row to PostgreSQL TenderDB fields."""
    # Generate a new UUID for the tender
    new_id = str(uuid.uuid4())

    # Get title from tender_title or title field
    title = row.get("tender_title") or row.get("title") or ""

    # Parse dates
    published_at = parse_migration_date(row.get("published_at", ""))
    deadline = parse_migration_date(row.get("deadline", ""))

    # Parse JSON fields
    critical_dates = parse_json_field(row.get("critical_dates"))
    tender_documents = parse_json_field(row.get("tender_documents"))
    tender_fee_details = parse_json_field(row.get("tender_fee_details"))
    emd_fee_details = parse_json_field(row.get("emd_fee_details"))
    work_item_details = parse_json_field(row.get("work_item_details"))
    tender_inviting_authority = parse_json_field(row.get("tender_inviting_authority"))
    covers_information = parse_json_field(row.get("covers_information"))
    payment_instruments = parse_json_field(row.get("payment_instruments"))
    tags = parse_json_field(row.get("tags"))

    # Build the mapped data
    mapped = {
        "id": new_id,
        "tender_id": row.get("tender_id"),
        "tender_reference_number": row.get("tender_reference_number"),
        "title": title,
        "authority": row.get("authority_name") or row.get("authority") or row.get("organisation_name"),
        "source": row.get("portal"),
        "source_url": row.get("url"),
        "tender_type": row.get("tender_type"),
        "tender_category": row.get("tender_category"),
        "estimated_value": parse_float_value(row.get("estimated_value")),
        "currency": "INR",
        "published_at": published_at,
        "deadline": deadline,
        "state": row.get("state"),
        "category": row.get("category"),
        "summary": row.get("summary"),
        "tags": tags,

        # Organisation fields
        "organisation_chain": row.get("organisation_chain"),

        # Additional detail fields
        "general_technical_evaluation_allowed": str(row.get("general_technical_evaluation_allowed")) if row.get("general_technical_evaluation_allowed") is not None else None,
        "payment_mode": row.get("payment_mode"),
        "withdrawal_allowed": str(row.get("withdrawal_allowed")) if row.get("withdrawal_allowed") is not None else None,
        "form_of_contract": row.get("form_of_contract"),
        "no_of_covers": str(row.get("no_of_covers")) if row.get("no_of_covers") is not None else None,
        "itemwise_technical_evaluation_allowed": str(row.get("itemwise_technical_evaluation_allowed")) if row.get("itemwise_technical_evaluation_allowed") is not None else None,
        "allow_two_stage_bidding": str(row.get("allow_two_stage_bidding")) if row.get("allow_two_stage_bidding") is not None else None,

        # JSON fields
        "payment_instruments": payment_instruments,
        "covers_information": covers_information,
        "tender_fee_details": tender_fee_details,
        "emd_fee_details": emd_fee_details,
        "work_item_details": work_item_details,
        "critical_dates": critical_dates,
        "tender_documents": tender_documents,
        "tender_inviting_authority": tender_inviting_authority,

        # Metadata
        "scraped_at": datetime.utcnow(),
        "scraper_version": "migration-1.0",
        "search_term_used": "migrated",
    }

    return mapped


def upload_document_to_s3(
    tender_uuid: str,
    file_path: Path,
    dry_run: bool = False
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Upload a document file to S3.

    Returns:
        Tuple of (success, s3_key, s3_url)
    """
    if not file_path.exists():
        logger.warning(f"Document file not found: {file_path}")
        return False, None, None

    filename = file_path.name
    s3_key = f"tenders/{tender_uuid}/{filename}"
    content_type = get_mime_type(filename)

    if dry_run:
        logger.info(f"  [DRY-RUN] Would upload: {filename} → s3://{s3_key}")
        return True, s3_key, f"https://tenderhub-documents.s3.amazonaws.com/{s3_key}"

    try:
        with open(file_path, 'rb') as f:
            file_data = f.read()

        success, result = upload_to_s3(
            file_data=file_data,
            s3_key=s3_key,
            content_type=content_type,
            metadata={
                "tender_id": tender_uuid,
                "original_filename": filename,
                "migrated_at": datetime.utcnow().isoformat()
            }
        )

        if success:
            logger.info(f"  Uploaded: {filename} → {s3_key}")
            return True, s3_key, result
        else:
            logger.error(f"  Failed to upload {filename}: {result}")
            return False, None, None

    except Exception as e:
        logger.error(f"  Error uploading {filename}: {e}")
        return False, None, None


def migrate_tender_documents(
    session,
    tender_uuid: str,
    portal: str,
    document_folder: str,
    dry_run: bool = False
) -> Tuple[int, int, List[str]]:
    """
    Migrate documents for a tender.

    Returns:
        Tuple of (success_count, total_count, uploaded_s3_keys)
    """
    portal_folder = PORTAL_FOLDER_MAP.get(portal)
    if not portal_folder:
        logger.warning(f"Unknown portal: {portal}")
        return 0, 0, []

    doc_dir = Path(DOCUMENT_BASE_PATH) / portal_folder / document_folder
    if not doc_dir.exists():
        logger.info(f"  Document directory not found: {doc_dir}")
        return 0, 0, []

    files = list(doc_dir.iterdir())
    if not files:
        return 0, 0, []

    success_count = 0
    uploaded_s3_keys = []

    for idx, file_path in enumerate(files):
        if not file_path.is_file():
            continue

        # Upload to S3
        success, s3_key, s3_url = upload_document_to_s3(tender_uuid, file_path, dry_run)

        if success:
            uploaded_s3_keys.append(s3_key)

            if not dry_run:
                # Create TenderDocumentDB record
                file_size = file_path.stat().st_size
                doc_record = TenderDocumentDB(
                    tender_id=tender_uuid,
                    document_type=get_document_type(file_path.name),
                    filename=file_path.name,
                    mime_type=get_mime_type(file_path.name),
                    file_size=file_size,
                    file_data=None,  # Not storing binary in DB
                    s3_key=s3_key,
                    s3_url=s3_url,
                    migrated_to_s3=True,
                    display_order=idx
                )
                session.add(doc_record)

            success_count += 1

    return success_count, len(files), uploaded_s3_keys


def cleanup_s3_files(s3_keys: List[str]) -> None:
    """Delete uploaded S3 files on rollback."""
    for s3_key in s3_keys:
        try:
            delete_from_s3(s3_key)
            logger.info(f"  Cleaned up S3 file: {s3_key}")
        except Exception as e:
            logger.warning(f"  Failed to cleanup S3 file {s3_key}: {e}")


def migrate_tenders(dry_run: bool = False, skip_documents: bool = False) -> Dict[str, Any]:
    """
    Main migration function.

    Returns:
        Migration statistics dictionary.
    """
    stats = {
        "started_at": datetime.utcnow().isoformat(),
        "dry_run": dry_run,
        "skip_documents": skip_documents,
        "total_tenders": 0,
        "migrated_tenders": 0,
        "skipped_duplicates": 0,
        "failed_tenders": 0,
        "total_documents": 0,
        "uploaded_documents": 0,
        "failed_documents": 0,
        "errors": []
    }

    # Load progress
    progress = load_progress()
    if not progress.get("started_at"):
        progress["started_at"] = stats["started_at"]

    logger.info("=" * 60)
    logger.info("Migration: NewScraper SQLite → Render PostgreSQL + S3")
    logger.info("=" * 60)
    logger.info(f"Dry run: {dry_run}")
    logger.info(f"Skip documents: {skip_documents}")
    logger.info(f"SQLite source: {SQLITE_PATH}")
    logger.info(f"Documents source: {DOCUMENT_BASE_PATH}")
    logger.info("")

    # Connect to SQLite
    sqlite_conn = sqlite3.connect(SQLITE_PATH)
    sqlite_conn.row_factory = sqlite3.Row
    sqlite_cursor = sqlite_conn.cursor()

    # Get PostgreSQL session
    if not dry_run:
        pg_session, pg_engine = get_postgresql_session()
        logger.info("Connected to PostgreSQL")
    else:
        pg_session = None
        pg_engine = None
        logger.info("[DRY-RUN] Skipping PostgreSQL connection")

    try:
        # Count total tenders
        sqlite_cursor.execute("SELECT COUNT(*) FROM tenders")
        stats["total_tenders"] = sqlite_cursor.fetchone()[0]
        logger.info(f"Total tenders in SQLite: {stats['total_tenders']}")

        # Fetch all tenders
        sqlite_cursor.execute("""
            SELECT * FROM tenders
            ORDER BY id
        """)

        batch_count = 0

        for row in sqlite_cursor:
            row_dict = dict(row)
            sqlite_id = row_dict.get("id")
            tender_id = row_dict.get("tender_id")
            source_url = row_dict.get("url")
            portal = row_dict.get("portal")
            document_folder = row_dict.get("document_folder")

            logger.info("-" * 40)
            logger.info(f"Processing SQLite ID {sqlite_id}: {tender_id}")

            # Check if already migrated (from progress file)
            if str(sqlite_id) in [str(x) for x in progress.get("migrated_ids", [])]:
                logger.info(f"  Already migrated (from progress), skipping")
                stats["skipped_duplicates"] += 1
                continue

            # Check for duplicates in PostgreSQL
            if not dry_run and check_duplicate(pg_session, tender_id, source_url):
                logger.info(f"  Duplicate found in PostgreSQL, skipping")
                stats["skipped_duplicates"] += 1
                progress["migrated_ids"].append(sqlite_id)
                continue

            # Track S3 uploads for potential rollback
            uploaded_s3_keys = []

            try:
                # Map fields
                mapped_data = map_sqlite_to_postgresql(row_dict)
                new_tender_uuid = mapped_data["id"]

                logger.info(f"  New UUID: {new_tender_uuid}")
                logger.info(f"  Title: {mapped_data.get('title', '')[:50]}...")
                logger.info(f"  Portal: {portal}")

                if not dry_run:
                    # Create tender record
                    tender_record = TenderDB(**mapped_data)
                    pg_session.add(tender_record)
                    pg_session.flush()  # Get ID without committing

                # Migrate documents
                if not skip_documents and document_folder:
                    doc_success, doc_total, uploaded_s3_keys = migrate_tender_documents(
                        pg_session,
                        new_tender_uuid,
                        portal,
                        document_folder,
                        dry_run
                    )
                    stats["total_documents"] += doc_total
                    stats["uploaded_documents"] += doc_success
                    stats["failed_documents"] += (doc_total - doc_success)
                    logger.info(f"  Documents: {doc_success}/{doc_total} uploaded")

                if not dry_run:
                    batch_count += 1

                    # Commit in batches
                    if batch_count >= BATCH_SIZE:
                        pg_session.commit()
                        logger.info(f"  Committed batch of {batch_count} tenders")
                        batch_count = 0

                stats["migrated_tenders"] += 1
                progress["migrated_ids"].append(sqlite_id)
                progress["last_sqlite_id"] = sqlite_id

                # Save progress periodically
                if stats["migrated_tenders"] % 10 == 0:
                    save_progress(progress)

            except Exception as e:
                logger.error(f"  Error migrating tender {sqlite_id}: {e}")
                stats["failed_tenders"] += 1
                stats["errors"].append({
                    "sqlite_id": sqlite_id,
                    "tender_id": tender_id,
                    "error": str(e)
                })

                if not dry_run:
                    # Rollback this tender
                    pg_session.rollback()

                    # Cleanup S3 files
                    if uploaded_s3_keys:
                        cleanup_s3_files(uploaded_s3_keys)

                continue

        # Commit remaining tenders
        if not dry_run and batch_count > 0:
            pg_session.commit()
            logger.info(f"Committed final batch of {batch_count} tenders")

        # Save final progress
        save_progress(progress)

    finally:
        sqlite_conn.close()
        if pg_session:
            pg_session.close()

    stats["completed_at"] = datetime.utcnow().isoformat()
    return stats


def verify_migration() -> Dict[str, Any]:
    """Verify the migration results."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("Verification")
    logger.info("=" * 60)

    verification = {
        "sqlite_count": 0,
        "postgresql_count": 0,
        "documents_count": 0,
        "sample_tenders": []
    }

    # Count SQLite tenders
    sqlite_conn = sqlite3.connect(SQLITE_PATH)
    cursor = sqlite_conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM tenders")
    verification["sqlite_count"] = cursor.fetchone()[0]
    sqlite_conn.close()

    # Count PostgreSQL tenders
    try:
        pg_session, _ = get_postgresql_session()
        verification["postgresql_count"] = pg_session.query(TenderDB).count()
        verification["documents_count"] = pg_session.query(TenderDocumentDB).filter(
            TenderDocumentDB.migrated_to_s3 == True
        ).count()

        # Get sample tenders
        samples = pg_session.query(TenderDB).limit(3).all()
        for sample in samples:
            verification["sample_tenders"].append({
                "id": sample.id,
                "tender_id": sample.tender_id,
                "title": sample.title[:50] if sample.title else "",
                "source": sample.source,
                "has_documents": pg_session.query(TenderDocumentDB).filter(
                    TenderDocumentDB.tender_id == sample.id
                ).count()
            })

        pg_session.close()
    except Exception as e:
        logger.error(f"Verification error: {e}")
        verification["error"] = str(e)

    logger.info(f"SQLite tenders: {verification['sqlite_count']}")
    logger.info(f"PostgreSQL tenders: {verification['postgresql_count']}")
    logger.info(f"Documents in S3: {verification['documents_count']}")

    if verification.get("sample_tenders"):
        logger.info("")
        logger.info("Sample migrated tenders:")
        for sample in verification["sample_tenders"]:
            logger.info(f"  - {sample['tender_id']}: {sample['title']}... ({sample['has_documents']} docs)")

    return verification


def print_report(stats: Dict[str, Any], verification: Dict[str, Any]) -> None:
    """Print final migration report."""
    print("")
    print("=" * 60)
    print("MIGRATION REPORT")
    print("=" * 60)
    print(f"Started:    {stats.get('started_at', 'N/A')}")
    print(f"Completed:  {stats.get('completed_at', 'N/A')}")
    print(f"Dry run:    {stats.get('dry_run', False)}")
    print("")
    print("Tenders:")
    print(f"  Total in source:     {stats.get('total_tenders', 0)}")
    print(f"  Successfully migrated: {stats.get('migrated_tenders', 0)}")
    print(f"  Skipped (duplicates): {stats.get('skipped_duplicates', 0)}")
    print(f"  Failed:              {stats.get('failed_tenders', 0)}")
    print("")
    print("Documents:")
    print(f"  Total found:        {stats.get('total_documents', 0)}")
    print(f"  Successfully uploaded: {stats.get('uploaded_documents', 0)}")
    print(f"  Failed:             {stats.get('failed_documents', 0)}")
    print("")
    print("Verification:")
    print(f"  SQLite count:       {verification.get('sqlite_count', 'N/A')}")
    print(f"  PostgreSQL count:   {verification.get('postgresql_count', 'N/A')}")
    print(f"  S3 documents:       {verification.get('documents_count', 'N/A')}")
    print("")

    if stats.get("errors"):
        print("Errors:")
        for error in stats["errors"][:5]:  # Show first 5 errors
            print(f"  - SQLite ID {error['sqlite_id']}: {error['error']}")
        if len(stats["errors"]) > 5:
            print(f"  ... and {len(stats['errors']) - 5} more errors")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Migrate tenders from NewScraper SQLite to PostgreSQL + S3"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview migration without making changes"
    )
    parser.add_argument(
        "--skip-documents",
        action="store_true",
        help="Skip document upload, migrate tenders only"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only run verification, skip migration"
    )

    args = parser.parse_args()

    if args.verify_only:
        verification = verify_migration()
        print_report({}, verification)
        return

    # Run migration
    stats = migrate_tenders(
        dry_run=args.dry_run,
        skip_documents=args.skip_documents
    )

    # Verify results (skip for dry run)
    if not args.dry_run:
        verification = verify_migration()
    else:
        verification = {"note": "Verification skipped for dry run"}

    # Print report
    print_report(stats, verification)

    # Save report to file
    report = {
        "stats": stats,
        "verification": verification
    }
    report_file = f"migration_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved to: {report_file}")


if __name__ == "__main__":
    main()
