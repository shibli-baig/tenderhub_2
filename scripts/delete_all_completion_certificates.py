#!/usr/bin/env python3
"""
Delete ALL completion certificates for ALL users.

Removes completion certificates from:
1. CertificateDB (new system with AI extraction) - all users
2. ProjectDB.documents['completion_certificate'] (old system) - all users
3. S3 storage (certificate files)
4. VectorDB, FAISS indices, local files (via certificate_processor cleanup)

Related records (TenderCertificateAttachmentDB, TenderCertificateMatchDB,
TenderMatchResultDB) are deleted via CASCADE when certificates are removed.

Usage:
    # Preview what would be deleted (dry run)
    python scripts/delete_all_completion_certificates.py --dry-run

    # Actually delete (requires confirmation)
    python scripts/delete_all_completion_certificates.py --confirm
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from sqlalchemy.orm import Session

from database import (
    SessionLocal,
    CertificateDB,
    ProjectDB,
)
from certificate_processor import certificate_processor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Optional S3 deletion (may not be configured locally)
try:
    from s3_utils import delete_from_s3
except ImportError:
    delete_from_s3 = None


def delete_certificates_for_all_users(db: Session, dry_run: bool = True) -> dict:
    """
    Delete all completion certificates from CertificateDB for all users.
    Returns stats dict.
    """
    certs = db.query(CertificateDB).all()
    total = len(certs)
    deleted = 0
    s3_deleted = 0
    errors = []

    logger.info(f"Found {total} certificate(s) in CertificateDB across all users")

    for cert in certs:
        cert_id = cert.id
        user_id = cert.user_id
        project_name = getattr(cert, "project_name", "?") or "?"

        if dry_run:
            logger.info(f"  [DRY RUN] Would delete certificate {cert_id} ({project_name})")
            if cert.s3_key:
                logger.info(f"    [DRY RUN] Would delete S3: {cert.s3_key}")
            deleted += 1
            continue

        try:
            # Delete from S3 if stored there
            if cert.s3_key and delete_from_s3:
                try:
                    if delete_from_s3(cert.s3_key):
                        s3_deleted += 1
                        logger.info(f"  Deleted S3: {cert.s3_key}")
                    else:
                        logger.warning(f"  Failed to delete S3: {cert.s3_key}")
                except Exception as e:
                    logger.warning(f"  S3 delete error for {cert.s3_key}: {e}")

            # Use certificate_processor for vectors, FAISS, local file, DB
            if certificate_processor.delete_certificate_with_cleanup(cert_id, user_id):
                deleted += 1
                logger.info(f"  Deleted certificate {cert_id} ({project_name})")
            else:
                errors.append(f"Certificate {cert_id}")

        except Exception as e:
            logger.error(f"  Error deleting certificate {cert_id}: {e}")
            errors.append(f"Certificate {cert_id}: {e}")

    return {
        "total": total,
        "deleted": deleted,
        "s3_deleted": s3_deleted,
        "errors": errors,
    }


def delete_project_completion_certificates(db: Session, dry_run: bool = True) -> dict:
    """
    Remove completion_certificate from ProjectDB.documents for all projects.
    Returns stats dict.
    """
    projects = db.query(ProjectDB).all()
    affected = 0
    file_count = 0

    for project in projects:
        if not project.documents or "completion_certificate" not in project.documents:
            continue

        certs = project.documents["completion_certificate"]
        if not certs:
            continue

        if dry_run:
            logger.info(
                f"  [DRY RUN] Would remove {len(certs)} cert(s) from project {project.id} ({project.project_name})"
            )
            affected += 1
            file_count += len(certs)
            continue

        # Delete local files if they exist
        for cert_path in certs:
            try:
                if isinstance(cert_path, str) and os.path.exists(cert_path):
                    os.remove(cert_path)
                    file_count += 1
                    logger.info(f"  Deleted old certificate file: {cert_path}")
            except Exception as e:
                logger.warning(f"  Failed to delete {cert_path}: {e}")

        # Remove key from documents
        documents_copy = dict(project.documents)
        del documents_copy["completion_certificate"]
        project.documents = documents_copy
        affected += 1

    if not dry_run and affected > 0:
        db.commit()

    return {"affected_projects": affected, "files_deleted": file_count}


def main():
    parser = argparse.ArgumentParser(
        description="Delete ALL completion certificates for ALL users"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be deleted without making changes",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually perform the deletion (required for real run)",
    )
    args = parser.parse_args()

    if not args.dry_run and not args.confirm:
        logger.error("Must specify --dry-run to preview OR --confirm to execute.")
        logger.error("  python scripts/delete_all_completion_certificates.py --dry-run")
        logger.error("  python scripts/delete_all_completion_certificates.py --confirm")
        sys.exit(1)

    dry_run = args.dry_run
    if dry_run:
        logger.info("=" * 60)
        logger.info("DRY RUN - No changes will be made")
        logger.info("=" * 60)

    db = SessionLocal()
    try:
        # Part 1: CertificateDB
        logger.info("\n--- CertificateDB (new system) ---")
        cert_stats = delete_certificates_for_all_users(db, dry_run=dry_run)
        logger.info(
            f"CertificateDB: {cert_stats['deleted']}/{cert_stats['total']} processed"
        )
        if cert_stats.get("s3_deleted"):
            logger.info(f"  S3 files deleted: {cert_stats['s3_deleted']}")
        if cert_stats.get("errors"):
            logger.warning(f"  Errors: {len(cert_stats['errors'])}")

        # Part 2: ProjectDB.documents
        logger.info("\n--- ProjectDB.documents (old system) ---")
        proj_stats = delete_project_completion_certificates(db, dry_run=dry_run)
        logger.info(
            f"Projects: {proj_stats['affected_projects']} affected, "
            f"{proj_stats['files_deleted']} old cert files"
        )

        # Summary
        total_certs = cert_stats["deleted"] + proj_stats["files_deleted"]
        logger.info("\n" + "=" * 60)
        if dry_run:
            logger.info(
                f"DRY RUN complete. Would delete {cert_stats['total']} certificates "
                f"and clear {proj_stats['affected_projects']} projects."
            )
            logger.info("Run with --confirm to execute.")
        else:
            logger.info(
                f"Done. Deleted {cert_stats['deleted']} certificates, "
                f"cleared {proj_stats['affected_projects']} projects."
            )
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        db.rollback()
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()
