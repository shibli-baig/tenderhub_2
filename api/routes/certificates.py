from fastapi import APIRouter, Request, Form, Depends, HTTPException, File, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, cast, String, desc, func, text
from sqlalchemy.dialects.postgresql import JSONB
from datetime import datetime
from typing import Optional, List, Dict, Any
import uuid
import os
import logging
import zipfile
import shutil
import hashlib
from pathlib import Path
from io import BytesIO

from database import CertificateDB, VectorDB, BulkUploadBatchDB
from core.dependencies import get_db, get_current_user, require_company_details
from core.openai_wrapper import OpenAIWrapper, OpenAIServiceError
from core.s3_storage import s3_manager, upload_to_s3
from certificate_processor import certificate_processor
from certificate_queue import (
    enqueue_certificate,
    get_queue_status,
    get_failed_tasks,
    retry_failed_task,
    clear_failed_tasks
)
import re
import json
import asyncio
import base64

router = APIRouter()
templates = Jinja2Templates(directory="templates")
logger = logging.getLogger(__name__)

def sanitize_filename(filename: str) -> str:
    """Remove problematic characters from filenames before saving."""
    if not filename:
        return "certificate"
    safe = re.sub(r'[^A-Za-z0-9._-]+', '_', filename)
    safe = safe.strip('_')
    return safe or "certificate"


def save_file_to_batch(dest_dir: Path, original_filename: str, content: bytes) -> str:
    """Persist uploaded certificate bytes to the batch-specific directory."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    unique_name = f"{uuid.uuid4()}_{sanitize_filename(original_filename)}"
    full_path = dest_dir / unique_name
    with open(full_path, "wb") as buffer:
        buffer.write(content)
    return str(full_path)


def compute_file_hash(file_content: bytes) -> str:
    """Compute SHA256 hash of file content for duplicate detection."""
    return hashlib.sha256(file_content).hexdigest()


def check_duplicate_certificate(db: Session, user_id: str, file_hash: str) -> Optional[CertificateDB]:
    """
    Check if a certificate with the same file hash already exists for this user.

    Returns:
        Existing certificate if found, None otherwise
    """
    return db.query(CertificateDB).filter(
        and_(
            CertificateDB.user_id == user_id,
            CertificateDB.file_hash == file_hash,
            CertificateDB.processing_status.in_(['completed', 'processing'])
        )
    ).first()


def parse_consultancy_fee(fee_str: str) -> float:
    """
    Parse consultancy fee string to numeric value.

    Handles formats like:
    - "₹5,00,000"
    - "Rs. 2.5 Crore"
    - "10 Lakh"
    - "5000000"

    Returns:
        Float value in rupees, or 0 if parsing fails
    """
    if not fee_str:
        return 0.0

    try:
        # Convert to string and lowercase
        fee_str = str(fee_str).lower()

        # Remove currency symbols, commas, and extra spaces
        clean = re.sub(r'[₹rs.\s,]', '', fee_str)

        # Extract numeric part
        numeric_match = re.search(r'[\d.]+', clean)
        if not numeric_match:
            return 0.0

        numeric_value = float(numeric_match.group())

        # Handle Crore/Lakh multipliers
        if 'crore' in fee_str or 'cr' in fee_str:
            return numeric_value * 10000000  # 1 Crore = 10 million
        elif 'lakh' in fee_str or 'lac' in fee_str or 'l' in fee_str:
            # Be careful with 'l' - only if it's standalone
            if re.search(r'\bl\b', fee_str):
                return numeric_value * 100000  # 1 Lakh = 100 thousand
            elif 'lakh' in fee_str or 'lac' in fee_str:
                return numeric_value * 100000

        return numeric_value

    except Exception as e:
        logger.warning(f"Failed to parse consultancy fee '{fee_str}': {e}")
        return 0.0


def format_currency_range(min_val: float, max_val: float) -> str:
    """Format currency range for display."""
    def format_inr(val):
        if val >= 10000000:  # Crore
            return f"₹{val/10000000:.1f} Cr"
        elif val >= 100000:  # Lakh
            return f"₹{val/100000:.1f} L"
        elif val >= 1000:  # Thousand
            return f"₹{val/1000:.0f}K"
        else:
            return f"₹{val:,.0f}"

    return f"{format_inr(min_val)} - {format_inr(max_val)}"


def calculate_fee_ranges(db: Session, user_id: str) -> dict:
    """
    Calculate 8 equal linear divisions for consultancy fees.

    Returns:
        Dictionary with min, max, and 8 range divisions
    """
    try:
        # Get all fee values for this user
        fees_raw = db.query(CertificateDB.consultancy_fee_inr).filter(
            CertificateDB.user_id == user_id,
            CertificateDB.consultancy_fee_inr.isnot(None)
        ).all()

        # Parse to numeric values
        fees = [parse_consultancy_fee(f[0]) for f in fees_raw if f[0]]
        fees = [f for f in fees if f > 0]  # Remove zeros

        if not fees:
            return {
                "min": 0,
                "max": 0,
                "divisions": []
            }

        min_fee = min(fees)
        max_fee = max(fees)

        # Create 8 equal linear divisions
        step = (max_fee - min_fee) / 8 if max_fee > min_fee else 0
        divisions = []

        for i in range(8):
            range_min = min_fee + (step * i)
            range_max = min_fee + (step * (i + 1)) if i < 7 else max_fee

            divisions.append({
                "label": format_currency_range(range_min, range_max),
                "min": range_min,
                "max": range_max,
                "index": i
            })

        return {
            "min": min_fee,
            "max": max_fee,
            "divisions": divisions
        }

    except Exception as e:
        logger.error(f"Error calculating fee ranges: {e}")
        return {
            "min": 0,
            "max": 0,
            "divisions": []
        }


def get_distinct_json_values(db: Session, field_name: str, user_id: str) -> List[str]:
    """
    Get distinct values from a JSONB array field.

    Args:
        db: Database session
        field_name: Name of the JSONB field (e.g., 'sectors', 'services_rendered')
        user_id: User ID to filter by

    Returns:
        List of distinct string values
    """
    try:
        # Get all certificates for this user
        certificates = db.query(CertificateDB).filter(
            CertificateDB.user_id == user_id
        ).all()

        # Extract and flatten values from JSONB field
        values_set = set()
        for cert in certificates:
            field_value = getattr(cert, field_name, None)
            if field_value:
                # Handle both JSON string and Python list
                if isinstance(field_value, str):
                    try:
                        parsed = json.loads(field_value)
                        if isinstance(parsed, list):
                            values_set.update([str(v) for v in parsed if v])
                    except json.JSONDecodeError:
                        pass
                elif isinstance(field_value, list):
                    values_set.update([str(v) for v in field_value if v])

        return sorted(list(values_set))

    except Exception as e:
        logger.error(f"Error getting distinct values for {field_name}: {e}")
        return []


def get_distinct_values(db: Session, field_name: str, user_id: str) -> List[str]:
    """
    Get distinct values from a regular text field.

    Args:
        db: Database session
        field_name: Name of the field
        user_id: User ID to filter by

    Returns:
        List of distinct non-null values
    """
    try:
        model_field = getattr(CertificateDB, field_name)
        values = db.query(model_field).filter(
            CertificateDB.user_id == user_id,
            model_field.isnot(None),
            model_field != ''
        ).distinct().all()

        return sorted([v[0] for v in values if v[0]])

    except Exception as e:
        logger.error(f"Error getting distinct values for {field_name}: {e}")
        return []


@router.get("/manage_certificates", response_class=HTMLResponse)
@require_company_details
async def manage_certificates_page(request: Request, db: Session = Depends(get_db)):
    """Manage certificates page (upload and search)."""
    current_user = get_current_user(request, db)
    if not current_user:
        return RedirectResponse(url="/login", status_code=302)

    return templates.TemplateResponse("manage_certificates.html", {
        "request": request,
        "current_user": current_user
    })


@router.get("/certificates/search", response_class=HTMLResponse)
@require_company_details
async def certificates_search_page(request: Request, db: Session = Depends(get_db)):
    """Certificate search page."""
    current_user = get_current_user(request, db)
    if not current_user:
        return RedirectResponse(url="/login", status_code=302)

    return templates.TemplateResponse("certificate_search.html", {
        "request": request,
        "current_user": current_user
    })


@router.get("/certificate/{certificate_id}", response_class=HTMLResponse)
@require_company_details
async def certificate_detail_page(request: Request, certificate_id: str, db: Session = Depends(get_db)):
    """Certificate detail page."""
    current_user = get_current_user(request, db)
    if not current_user:
        return RedirectResponse(url="/login", status_code=302)

    # Get the certificate
    certificate = db.query(CertificateDB).filter(
        and_(CertificateDB.id == certificate_id, CertificateDB.user_id == current_user.id)
    ).first()

    if not certificate:
        raise HTTPException(status_code=404, detail="Certificate not found")

    return templates.TemplateResponse("certificate_detail.html", {
        "request": request,
        "current_user": current_user,
        "certificate": certificate
    })


@router.post("/api/certificates/upload")
async def upload_certificates_bulk(
    request: Request,
    files: List[UploadFile] = File(default=[]),
    batch_name: Optional[str] = Form(default=None),
    db: Session = Depends(get_db)
):
    """
    Upload and process multiple certificate documents.
    Supports:
    - Multiple individual files (PDF, JPG, PNG)
    - Zip files containing multiple certificates
    - Folder uploads (sent as multiple files by browser)
    """
    try:
        # Log request details
        logger.info(f"Upload endpoint called")
        logger.info(f"Request method: {request.method}")
        logger.info(f"Request headers: {dict(request.headers)}")
        logger.info(f"Content-Type: {request.headers.get('content-type')}")
        logger.info(f"Files parameter count: {len(files) if files else 0}")

        if files:
            for idx, file in enumerate(files):
                logger.info(f"File {idx}: {file.filename}, content_type: {file.content_type}")
        else:
            logger.warning("No files in files parameter!")

            # Try to parse form manually
            try:
                form = await request.form()
                logger.info(f"Form keys: {list(form.keys())}")
                for key in form.keys():
                    value = form[key]
                    if hasattr(value, 'filename'):
                        logger.info(f"Found file in form[{key}]: {value.filename}")
                    else:
                        logger.info(f"Form[{key}]: {value}")
            except Exception as form_error:
                logger.error(f"Could not parse form: {form_error}")

        current_user = get_current_user(request, db)
        if not current_user:
            logger.error("No current user found")
            raise HTTPException(status_code=401, detail="Authentication required")

        # Check company details
        from core.dependencies import user_has_complete_company_details
        if not user_has_complete_company_details(current_user.id, db):
            logger.error(f"User {current_user.id} has incomplete company details")
            raise HTTPException(status_code=403, detail="Company details required")

        logger.info(f"User {current_user.id} authenticated successfully")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in upload endpoint initialization: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    # Create uploads directory
    os.makedirs("uploads/certificates", exist_ok=True)

    allowed_extensions = {'.pdf', '.jpg', '.jpeg', '.png', '.zip'}
    files_to_process: List[Dict[str, Any]] = []
    duplicates_detected: List[Dict[str, str]] = []
    contains_zip = any((uploaded_file.filename or "").lower().endswith(".zip") for uploaded_file in files)

    # Create batch-specific directory up front to stage files
    batch_id = str(uuid.uuid4())
    batch_dir = Path("uploads") / "certificates" / current_user.id / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Collect all files (extract zips if present) and stage them
        for uploaded_file in files:
            filename = uploaded_file.filename or "unknown"
            extension = Path(filename).suffix.lower()

            if extension not in allowed_extensions:
                logger.warning(f"Skipping unsupported file: {filename}")
                continue

            file_content = await uploaded_file.read()

            if not file_content:
                logger.warning(f"Uploaded file {filename} is empty. Skipping.")
                continue

            if extension == '.zip':
                logger.info(f"Extracting zip file: {filename}")
                extracted_files = await extract_certificates_from_zip(
                    file_content,
                    batch_dir,
                    current_user.id,
                    db
                )
                # Separate duplicates from files to process
                for info in extracted_files:
                    if info.get('is_duplicate') or info.get('duplicate_of'):
                        duplicates_detected.append({
                            'filename': info['filename'],
                            'duplicate_of': info['duplicate_of'],
                            'existing_project_name': info.get('existing_project_name')
                        })
                    else:
                        files_to_process.append(info)
            else:
                file_hash = compute_file_hash(file_content)
                file_size = len(file_content)

                existing_cert = check_duplicate_certificate(db, current_user.id, file_hash)
                if existing_cert:
                    # SKIP duplicate - do not process or save file
                    duplicates_detected.append({
                        'filename': filename,
                        'duplicate_of': existing_cert.id,
                        'existing_project_name': existing_cert.project_name
                    })
                    logger.info(f"⏭️ Skipping duplicate certificate: {filename} (matches existing: {existing_cert.id})")
                    continue  # Skip this file entirely

                # Only save and queue non-duplicate files
                file_path = save_file_to_batch(batch_dir, filename, file_content)

                files_to_process.append({
                    'file_path': file_path,
                    'filename': filename,
                    'file_hash': file_hash,
                    'file_size': file_size
                })

        if not files_to_process:
            shutil.rmtree(batch_dir, ignore_errors=True)
            # Check if all files were duplicates
            if duplicates_detected:
                duplicate_names = [d['filename'] for d in duplicates_detected]
                raise HTTPException(
                    status_code=400,
                    detail=f"All {len(duplicates_detected)} file(s) are duplicates and already exist in your certificate vault: {', '.join(duplicate_names[:5])}{'...' if len(duplicate_names) > 5 else ''}"
                )
            raise HTTPException(
                status_code=400,
                detail="No valid certificate files found. Supported formats: PDF, JPG, PNG, ZIP"
            )

        # Step 2: Create batch record
        upload_type = 'zip' if contains_zip else 'files'
        skipped_duplicate_count = len(duplicates_detected)

        batch = BulkUploadBatchDB(
            id=batch_id,
            user_id=current_user.id,
            batch_name=batch_name or f"Upload {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
            upload_type=upload_type,
            total_files=len(files_to_process),
            skipped_count=skipped_duplicate_count,
            status='queued',
            created_at=datetime.utcnow()
        )
        db.add(batch)
        db.commit()

        logger.info(f"Created batch {batch_id} with {len(files_to_process)} files staged for processing")

        # Step 4: Enqueue all files for background processing
        task_ids = []
        for file_info in files_to_process:
            task_id = enqueue_certificate(
                user_id=current_user.id,
                file_path=file_info['file_path'],
                filename=file_info['filename'],
                batch_id=batch_id,
                file_hash=file_info.get('file_hash'),
                file_size=file_info.get('file_size')
            )
            task_ids.append(task_id)

        logger.info(f"Enqueued {len(task_ids)} certificate processing tasks")

        message_parts = []
        if len(files_to_process) > 0:
            message_parts.append(f"{len(files_to_process)} new certificate(s) queued for processing")
        if skipped_duplicate_count > 0:
            message_parts.append(f"{skipped_duplicate_count} duplicate(s) skipped (already exist)")

        return {
            "message": ", ".join(message_parts) if message_parts else "No files to process",
            "batch_id": batch_id,
            "total_files": len(files_to_process),
            "duplicates_skipped": skipped_duplicate_count,
            "task_ids": task_ids,
            "skipped_duplicates": duplicates_detected,
            "skipped_files": skipped_duplicate_count
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Certificate bulk upload failed: {e}")
        # Clean up any saved files on error
        for file_info in files_to_process:
            try:
                if os.path.exists(file_info['file_path']):
                    os.remove(file_info['file_path'])
            except:
                pass
        shutil.rmtree(batch_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    finally:
        if not files_to_process:
            shutil.rmtree(batch_dir, ignore_errors=True)


async def extract_certificates_from_zip(
    zip_content: bytes,
    dest_dir: Path,
    user_id: str,
    db: Session
) -> List[Dict[str, Any]]:
    """
    Extract certificate files from a zip archive, stage them, and return metadata for processing.
    """
    files: List[Dict[str, Any]] = []
    allowed_extensions = {'.pdf', '.jpg', '.jpeg', '.png'}

    try:
        with zipfile.ZipFile(BytesIO(zip_content)) as zip_ref:
            bad_file = zip_ref.testzip()
            if bad_file:
                raise ValueError(f"Corrupted file in zip: {bad_file}")

            for zip_info in zip_ref.infolist():
                if zip_info.is_dir():
                    continue

                original_filename = os.path.basename(zip_info.filename)
                file_extension = Path(original_filename).suffix.lower()

                if file_extension not in allowed_extensions:
                    logger.debug(f"Skipping non-certificate file in zip: {original_filename}")
                    continue

                file_bytes = zip_ref.read(zip_info.filename)
                if not file_bytes:
                    continue

                file_hash = compute_file_hash(file_bytes)
                file_size = len(file_bytes)
                existing_cert = check_duplicate_certificate(db, user_id, file_hash)

                if existing_cert:
                    # SKIP duplicate - mark as duplicate but don't save file
                    files.append({
                        'file_path': None,
                        'filename': original_filename,
                        'file_hash': file_hash,
                        'file_size': file_size,
                        'duplicate_of': existing_cert.id,
                        'existing_project_name': existing_cert.project_name,
                        'is_duplicate': True
                    })
                    logger.info(f"⏭️ Skipping duplicate from zip: {original_filename} (matches existing: {existing_cert.id})")
                    continue

                permanent_path = save_file_to_batch(dest_dir, original_filename, file_bytes)

                files.append({
                    'file_path': permanent_path,
                    'filename': original_filename,
                    'file_hash': file_hash,
                    'file_size': file_size,
                    'duplicate_of': None,
                    'is_duplicate': False
                })

                logger.info(f"Extracted from zip: {original_filename} -> {permanent_path}")

    except zipfile.BadZipFile:
        raise ValueError("Invalid or corrupted zip file")
    except Exception as e:
        # Clean up staged files on error
        for file_info in files:
            try:
                if os.path.exists(file_info['file_path']):
                    os.remove(file_info['file_path'])
            except OSError:
                pass
        raise ValueError(f"Failed to extract zip file: {str(e)}")

    return files


@router.get("/api/certificates")
@require_company_details
async def get_certificates(
    request: Request,
    search: str = "",
    client: str = "",
    location: str = "",
    page: int = 1,
    per_page: int = 10,
    db: Session = Depends(get_db)
):
    """Get list of certificates with filters."""
    current_user = get_current_user(request, db)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Build query
    query = db.query(CertificateDB).filter(CertificateDB.user_id == current_user.id)

    # Apply filters
    if search:
        search_term = f"%{search}%"
        query = query.filter(
            (CertificateDB.certificate_name.ilike(search_term)) |
            (CertificateDB.certificate_description.ilike(search_term))
        )

    if client:
        query = query.filter(CertificateDB.client_name.ilike(f"%{client}%"))

    if location:
        query = query.filter(CertificateDB.work_location.ilike(f"%{location}%"))

    # Get total count
    total = query.count()

    # Apply pagination
    certificates = query.order_by(CertificateDB.created_at.desc()).offset((page - 1) * per_page).limit(per_page).all()

    return {
        "certificates": certificates,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page
    }


@router.get("/api/certificates/filter-options")
@require_company_details
async def get_certificate_filter_options(
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Return all available filter options based on user's certificates.

    This endpoint dynamically generates filter options from actual database values,
    including:
    - Distinct values for categorical filters (clients, locations, services, etc.)
    - Min/max ranges for numerical filters (project value, confidence score)
    - 8 equal linear divisions for consultancy fees
    - Date ranges for temporal filters
    """
    from sqlalchemy import func

    current_user = get_current_user(request, db)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    logger.info(f"📊 Fetching filter options for user {current_user.email}")

    try:
        # Get distinct values for categorical filters
        clients = get_distinct_values(db, 'client_name', current_user.id)
        locations = get_distinct_values(db, 'location', current_user.id)
        funding_agencies = get_distinct_values(db, 'funding_agency', current_user.id)

        # Get distinct values from JSONB fields
        services = get_distinct_json_values(db, 'services_rendered', current_user.id)
        sectors = get_distinct_json_values(db, 'sectors', current_user.id)
        sub_sectors = get_distinct_json_values(db, 'sub_sectors', current_user.id)

        # Calculate consultancy fee ranges (8 equal divisions)
        fee_ranges = calculate_fee_ranges(db, current_user.id)

        # Get project value min/max
        value_stats = db.query(
            func.min(CertificateDB.project_value),
            func.max(CertificateDB.project_value)
        ).filter(
            CertificateDB.user_id == current_user.id,
            CertificateDB.project_value.isnot(None)
        ).first()

        # Get date ranges
        date_stats = db.query(
            func.min(CertificateDB.completion_date),
            func.max(CertificateDB.completion_date)
        ).filter(
            CertificateDB.user_id == current_user.id,
            CertificateDB.completion_date.isnot(None)
        ).first()

        response = {
            "success": True,
            "clients": clients,
            "locations": locations,
            "services": services,
            "sectors": sectors,
            "sub_sectors": sub_sectors,
            "funding_agencies": funding_agencies,
            "roles": ["Lead Consultant", "JV Partner", "Consortium"],
            "processing_statuses": ["completed", "processing", "failed", "duplicate"],
            "ranges": {
                "consultancy_fee": fee_ranges,
                "project_value": {
                    "min": float(value_stats[0]) if value_stats[0] else 0,
                    "max": float(value_stats[1]) if value_stats[1] else 0
                },
                "completion_date": {
                    "min": date_stats[0].isoformat() if date_stats[0] else None,
                    "max": date_stats[1].isoformat() if date_stats[1] else None
                }
            }
        }

        logger.info(f"✅ Filter options generated: {len(clients)} clients, {len(locations)} locations, {len(services)} services")

        return response

    except Exception as e:
        logger.error(f"❌ Error fetching filter options: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching filter options: {str(e)}")


@router.get("/api/certificates/manual-clause/filter-options")
@require_company_details
async def get_manual_clause_filter_options(
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Return comprehensive filter options for Manual Clause filtering.

    Includes all fields from the regular filter options plus additional fields:
    - Duration (distinct values from duration field)
    - Certificate Number (distinct certificate numbers)
    - Role/Lead/JV (distinct values from role_lead_jv field)
    """
    from sqlalchemy import func

    current_user = get_current_user(request, db)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    logger.info(f"📊 Fetching manual clause filter options for user {current_user.email}")

    try:
        # Get distinct values for categorical filters
        clients = get_distinct_values(db, 'client_name', current_user.id)
        locations = get_distinct_values(db, 'location', current_user.id)
        funding_agencies = get_distinct_values(db, 'funding_agency', current_user.id)
        durations = get_distinct_values(db, 'duration', current_user.id)
        certificate_numbers = get_distinct_values(db, 'certificate_number', current_user.id)
        roles = get_distinct_values(db, 'role_lead_jv', current_user.id)

        # If no roles in DB, use default options
        if not roles:
            roles = ["Lead Consultant", "JV Partner", "Consortium", "Solo"]

        # Get distinct values from JSONB fields
        services = get_distinct_json_values(db, 'services_rendered', current_user.id)
        sectors = get_distinct_json_values(db, 'sectors', current_user.id)
        sub_sectors = get_distinct_json_values(db, 'sub_sectors', current_user.id)

        # Get consultancy fee min/max (for range slider)
        fee_stats = db.query(
            func.min(CertificateDB.consultancy_fee_numeric),
            func.max(CertificateDB.consultancy_fee_numeric)
        ).filter(
            CertificateDB.user_id == current_user.id,
            CertificateDB.consultancy_fee_numeric.isnot(None)
        ).first()

        # Get project value min/max
        value_stats = db.query(
            func.min(CertificateDB.project_value),
            func.max(CertificateDB.project_value)
        ).filter(
            CertificateDB.user_id == current_user.id,
            CertificateDB.project_value.isnot(None)
        ).first()

        # Get date ranges
        date_stats = db.query(
            func.min(CertificateDB.completion_date),
            func.max(CertificateDB.completion_date)
        ).filter(
            CertificateDB.user_id == current_user.id,
            CertificateDB.completion_date.isnot(None)
        ).first()

        response = {
            "success": True,
            "clients": clients,
            "locations": locations,
            "services": services,
            "sectors": sectors,
            "sub_sectors": sub_sectors,
            "funding_agencies": funding_agencies,
            "roles": roles,
            "durations": durations,
            "certificate_numbers": certificate_numbers,
            "consultancy_fee_range": {
                "min": float(fee_stats[0]) if fee_stats[0] else 0,
                "max": float(fee_stats[1]) if fee_stats[1] else 0
            },
            "project_value_range": {
                "min": float(value_stats[0]) if value_stats[0] else 0,
                "max": float(value_stats[1]) if value_stats[1] else 0
            },
            "completion_date_range": {
                "min": date_stats[0].isoformat() if date_stats[0] else None,
                "max": date_stats[1].isoformat() if date_stats[1] else None
            }
        }

        logger.info(f"✅ Manual clause filter options generated: {len(clients)} clients, {len(locations)} locations, {len(services)} services")

        return response

    except Exception as e:
        logger.error(f"❌ Error fetching manual clause filter options: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching filter options: {str(e)}")


def count_active_filters(filters: dict) -> int:
    """
    Count how many filters have non-empty values.
    """
    count = 0
    counted_filters = []  # Track which filters are counted

    # Multi-select filters
    for key in ['clients', 'locations', 'services', 'sectors', 'sub_sectors',
                'funding_agencies', 'roles', 'durations', 'certificate_numbers']:
        if filters.get(key) and len(filters[key]) > 0:
            count += 1
            counted_filters.append(key)
            logger.info(f"   ✅ Counted '{key}': {filters[key]} (count now: {count})")

    # Range filters (ONLY count if BOTH bounds are meaningfully constrained)
    if filters.get('consultancy_fee_range'):
        fee_range = filters['consultancy_fee_range']
        min_val = fee_range.get('min')
        max_val = fee_range.get('max')

        # STRICT: Only count if BOTH min AND max are set with meaningful constraints
        # Don't count if min=0 (default) even if max is set
        is_meaningful = False
        if min_val is not None and min_val > 0 and max_val is not None and max_val < 100000000:
            is_meaningful = True  # Both bounds are constrained
        elif min_val is not None and min_val > 0 and (max_val is None or max_val >= 100000000):
            is_meaningful = True  # Only min is constrained (max is at default or not set)

        if is_meaningful:
            count += 1
            counted_filters.append('consultancy_fee_range')
            logger.info(f"   ✅ Counted 'consultancy_fee_range': {fee_range} (count now: {count})")
        else:
            logger.info(f"   ⏭️ Skipped 'consultancy_fee_range' (min=0 or default range): {fee_range}")

    if filters.get('project_value_range'):
        value_range = filters['project_value_range']
        min_val = value_range.get('min')
        max_val = value_range.get('max')

        # STRICT: Only count if BOTH min AND max are set with meaningful constraints
        # Don't count if min=0 (default) even if max is set
        is_meaningful = False
        if min_val is not None and min_val > 0 and max_val is not None and max_val < 1000000000:
            is_meaningful = True  # Both bounds are constrained
        elif min_val is not None and min_val > 0 and (max_val is None or max_val >= 1000000000):
            is_meaningful = True  # Only min is constrained (max is at default or not set)

        if is_meaningful:
            count += 1
            counted_filters.append('project_value_range')
            logger.info(f"   ✅ Counted 'project_value_range': {value_range} (count now: {count})")
        else:
            logger.info(f"   ⏭️ Skipped 'project_value_range' (min=0 or default range): {value_range}")

    # Date range filter
    if filters.get('completion_date_range'):
        date_range = filters['completion_date_range']
        if date_range.get('start') or date_range.get('end'):
            count += 1
            counted_filters.append('completion_date_range')
            logger.info(f"   ✅ Counted 'completion_date_range': {date_range} (count now: {count})")

    logger.info(f"📊 Total filters counted: {count} → {counted_filters}")
    return count


# ============================================================================
# FUZZY MATCHING UTILITIES FOR MANUAL CLAUSE FILTERING
# ============================================================================

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein distance between two strings.
    Lower distance = more similar strings.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def similarity_score(s1: str, s2: str) -> float:
    """
    Calculate similarity score between two strings (0.0 to 1.0).
    1.0 = exact match, 0.0 = completely different.
    """
    if not s1 or not s2:
        return 0.0

    s1_lower = s1.lower().strip()
    s2_lower = s2.lower().strip()

    # Exact match (case-insensitive)
    if s1_lower == s2_lower:
        return 1.0

    # Substring match (one contains the other)
    if s1_lower in s2_lower or s2_lower in s1_lower:
        # Calculate containment ratio
        shorter = min(len(s1_lower), len(s2_lower))
        longer = max(len(s1_lower), len(s2_lower))
        return shorter / longer if longer > 0 else 0.0

    # Word-level matching (check if words from s1 appear in s2)
    s1_words = set(s1_lower.split())
    s2_words = set(s2_lower.split())
    if s1_words and s2_words:
        common_words = s1_words.intersection(s2_words)
        if common_words:
            # Calculate word overlap ratio
            word_overlap = len(common_words) / max(len(s1_words), len(s2_words))
            # Boost score if significant word overlap
            if word_overlap >= 0.5:
                return word_overlap * 0.9  # Slightly less than exact match

    # Use Levenshtein distance for fuzzy matching
    max_len = max(len(s1_lower), len(s2_lower))
    if max_len == 0:
        return 0.0

    distance = levenshtein_distance(s1_lower, s2_lower)
    similarity = 1.0 - (distance / max_len)
    
    # Normalize to ensure it's between 0 and 1
    return max(0.0, min(1.0, similarity))


def fuzzy_match_in_text(search_term: str, text: str, threshold: float = 0.3) -> tuple[bool, float, str]:
    """
    Check if search_term matches in text using fuzzy matching.
    
    Args:
        search_term: The term to search for
        text: The text to search in
        threshold: Minimum similarity score to consider a match (default: 0.3 = 30%)
    
    Returns:
        Tuple of (is_match: bool, best_score: float, matched_text: str)
    """
    if not search_term or not text:
        return False, 0.0, ""

    search_term_lower = search_term.lower().strip()
    text_lower = text.lower()

    # Direct substring match (highest priority)
    if search_term_lower in text_lower:
        # Find the actual matched text from original
        start_idx = text_lower.find(search_term_lower)
        matched_text = text[start_idx:start_idx + len(search_term)]
        return True, 1.0, matched_text

    # Word-level matching
    search_words = search_term_lower.split()
    text_words = text_lower.split()
    
    if search_words:
        # Check if all words from search term appear in text
        words_found = sum(1 for word in search_words if any(word in tw or tw in word for tw in text_words))
        word_match_ratio = words_found / len(search_words)
        
        if word_match_ratio >= 0.5:  # At least 50% of words match
            # Find matched words in original text
            matched_words = []
            for word in search_words:
                for tw in text_words:
                    if word in tw or tw in word:
                        matched_words.append(tw)
                        break
            matched_text = " ".join(matched_words[:3])  # Limit to first 3 words
            return True, word_match_ratio * 0.8, matched_text  # Slightly lower than exact match

    # Fuzzy similarity matching (check similarity with each word/phrase in text)
    best_score = 0.0
    best_match = ""
    
    # Try matching against individual words
    for word in text_words:
        score = similarity_score(search_term_lower, word)
        if score > best_score:
            best_score = score
            best_match = word
    
    # Try matching against phrases (2-3 word combinations)
    for i in range(len(text_words) - 1):
        phrase = " ".join(text_words[i:i+2])
        score = similarity_score(search_term_lower, phrase)
        if score > best_score:
            best_score = score
            best_match = phrase
    
    if best_score >= threshold:
        return True, best_score, best_match

    return False, best_score, ""


def fuzzy_match_in_list(search_terms: list, target_list: list, threshold: float = 0.3) -> tuple[bool, list, float]:
    """
    Check if any search_term matches any item in target_list using fuzzy matching.
    
    Args:
        search_terms: List of terms to search for
        target_list: List of items to search in (can be list of strings or list of dicts)
        threshold: Minimum similarity score to consider a match
    
    Returns:
        Tuple of (is_match: bool, matched_items: list, best_score: float)
    """
    if not search_terms or not target_list:
        return False, [], 0.0

    matched_items = []
    best_score = 0.0

    # Convert target_list to strings if needed
    target_strings = []
    for item in target_list:
        if isinstance(item, dict):
            # If it's a dict, try to get a string representation
            target_strings.append(str(item.get('value', item.get('label', str(item)))))
        else:
            target_strings.append(str(item))

    # Combine all target strings into a single searchable text
    combined_text = " ".join(target_strings).lower()

    for search_term in search_terms:
        is_match, score, matched_text = fuzzy_match_in_text(str(search_term), combined_text, threshold)
        if is_match:
            # Find the actual item(s) that matched
            search_term_lower = str(search_term).lower()
            for i, target_str in enumerate(target_strings):
                if search_term_lower in target_str.lower() or target_str.lower() in search_term_lower:
                    matched_items.append(target_list[i])
                    break
            best_score = max(best_score, score)

    return len(matched_items) > 0, matched_items, best_score


def fuzzy_match_multi_field(search_terms: list, certificate: CertificateDB, 
                           primary_field: str, related_fields: list = None,
                           threshold: float = 0.3) -> tuple[bool, float, dict]:
    """
    Perform fuzzy matching across multiple fields of a certificate.
    
    Args:
        search_terms: List of terms to search for
        certificate: CertificateDB object to search in
        primary_field: Primary field name to search (e.g., 'client_name', 'location')
        related_fields: List of additional field names to search (e.g., ['project_name', 'description'])
        threshold: Minimum similarity score to consider a match
    
    Returns:
        Tuple of (is_match: bool, best_score: float, match_details: dict)
    """
    if not search_terms:
        return False, 0.0, {}

    # Get primary field value
    primary_value = getattr(certificate, primary_field, None)
    if primary_value is None:
        primary_value = ""

    # Convert to string and handle lists/JSONB
    if isinstance(primary_value, list):
        primary_text = " ".join(str(v) for v in primary_value)
    elif isinstance(primary_value, dict):
        primary_text = " ".join(str(v) for v in primary_value.values())
    else:
        primary_text = str(primary_value)

    # Build searchable text from primary and related fields
    searchable_text = primary_text

    # Add related fields if provided
    if related_fields:
        for field_name in related_fields:
            field_value = getattr(certificate, field_name, None)
            if field_value:
                if isinstance(field_value, list):
                    searchable_text += " " + " ".join(str(v) for v in field_value)
                elif isinstance(field_value, dict):
                    searchable_text += " " + " ".join(str(v) for v in field_value.values())
                else:
                    searchable_text += " " + str(field_value)

    # Also search in full text if available (for maximum flexibility)
    full_text = getattr(certificate, 'full_verbatim_certificate', None)
    if full_text:
        searchable_text += " " + str(full_text)

    # Perform fuzzy matching
    best_match_found = False
    best_score = 0.0
    match_details = {
        'primary_field': primary_field,
        'primary_value': primary_text,
        'matched_terms': [],
        'match_scores': {}
    }

    for search_term in search_terms:
        is_match, score, matched_text = fuzzy_match_in_text(str(search_term), searchable_text, threshold)
        if is_match:
            best_match_found = True
            best_score = max(best_score, score)
            match_details['matched_terms'].append(str(search_term))
            match_details['match_scores'][str(search_term)] = round(score, 3)
            if matched_text and matched_text not in match_details.get('matched_texts', []):
                if 'matched_texts' not in match_details:
                    match_details['matched_texts'] = []
                match_details['matched_texts'].append(matched_text)

    return best_match_found, best_score, match_details


def calculate_compliance(certificate: CertificateDB, filters: dict) -> dict:
    """
    Calculate compliance score for a certificate against applied filters.

    Returns:
        {
            'total_filters': int,
            'filters_met': int,
            'compliance_percentage': float,
            'met_filters': List[str],
            'unmet_filters': List[str],
            'details': Dict[str, dict]
        }
    """
    total_filters = count_active_filters(filters)
    filters_met = 0
    met_filters = []
    unmet_filters = []
    details = {}

    # 🔍 DEBUG LOGGING: Show which filters are being checked (first certificate only to avoid spam)
    if not hasattr(calculate_compliance, '_logged_filters'):
        logger.info("=" * 100)
        logger.info("🔍 ACTIVE FILTERS BEING CHECKED AGAINST CERTIFICATES")
        logger.info("=" * 100)

        if filters.get('clients'):
            logger.info(f"✅ CLIENTS: {filters['clients']}")
        if filters.get('locations'):
            logger.info(f"✅ LOCATIONS: {filters['locations']}")
        if filters.get('services'):
            logger.info(f"✅ SERVICES: {filters['services']}")
        if filters.get('sectors'):
            logger.info(f"✅ SECTORS: {filters['sectors']}")
        if filters.get('sub_sectors'):
            logger.info(f"✅ SUB_SECTORS: {filters['sub_sectors']}")
        if filters.get('funding_agencies'):
            logger.info(f"✅ FUNDING_AGENCIES: {filters['funding_agencies']}")
        if filters.get('roles'):
            logger.info(f"✅ ROLES: {filters['roles']}")
        if filters.get('durations'):
            logger.info(f"✅ DURATIONS: {filters['durations']}")
        if filters.get('consultancy_fee_min') or filters.get('consultancy_fee_max'):
            logger.info(f"✅ CONSULTANCY_FEE_RANGE: ₹{filters.get('consultancy_fee_min', 0):,} - ₹{filters.get('consultancy_fee_max', 0):,}")
        if filters.get('project_value_min') or filters.get('project_value_max'):
            logger.info(f"✅ PROJECT_VALUE_RANGE: ₹{filters.get('project_value_min', 0):,} - ₹{filters.get('project_value_max', 0):,}")
        if filters.get('completion_date_start') or filters.get('completion_date_end'):
            logger.info(f"✅ COMPLETION_DATE: {filters.get('completion_date_start', 'N/A')} to {filters.get('completion_date_end', 'N/A')}")

        logger.info(f"📊 Total Active Filters: {total_filters}")
        logger.info("=" * 100)

        # Set flag to log only once per request
        calculate_compliance._logged_filters = True

    # Check Client Name (FUZZY MATCHING)
    if filters.get('clients') and len(filters['clients']) > 0:
        required = filters['clients']
        # Use fuzzy matching across client_name and related fields
        is_met, match_score, match_details = fuzzy_match_multi_field(
            required, certificate, 'client_name',
            related_fields=['project_name', 'description'],  # Also search in project name
            threshold=0.3  # 30% similarity threshold
        )
        matched = match_details.get('matched_texts', [None])[0] if is_met else None

        details['clients'] = {
            'required': required,
            'matched': matched,
            'met': is_met,
            'match_score': round(match_score, 3) if is_met else 0.0,
            'match_details': match_details
        }

        if is_met:
            filters_met += 1
            met_filters.append('clients')
        else:
            unmet_filters.append('clients')

    # Check Location (FUZZY MATCHING)
    if filters.get('locations') and len(filters['locations']) > 0:
        required = filters['locations']
        # Use fuzzy matching across location and related fields
        is_met, match_score, match_details = fuzzy_match_multi_field(
            required, certificate, 'location',
            related_fields=['project_name', 'description'],  # Also search in project name
            threshold=0.3
        )
        matched = match_details.get('matched_texts', [None])[0] if is_met else None

        details['locations'] = {
            'required': required,
            'matched': matched,
            'met': is_met,
            'match_score': round(match_score, 3) if is_met else 0.0,
            'match_details': match_details
        }

        if is_met:
            filters_met += 1
            met_filters.append('locations')
        else:
            unmet_filters.append('locations')

    # Check Services Rendered (FUZZY MATCHING - JSONB array - ANY required service matches - OR logic)
    if filters.get('services') and len(filters['services']) > 0:
        required = filters['services']
        cert_services = certificate.services_rendered or []
        
        # Use fuzzy matching - check if any required service matches any certificate service
        is_met, matched_items, best_score = fuzzy_match_in_list(required, cert_services, threshold=0.3)
        matched = matched_items if is_met else []

        details['services'] = {
            'required': required,
            'matched': matched,
            'met': is_met,
            'match_score': round(best_score, 3) if is_met else 0.0
        }

        if is_met:
            filters_met += 1
            met_filters.append('services')
        else:
            unmet_filters.append('services')

    # Check Sectors (FUZZY MATCHING - JSONB array - ANY required sector matches - OR logic)
    if filters.get('sectors') and len(filters['sectors']) > 0:
        required = filters['sectors']
        cert_sectors = certificate.sectors or []
        
        # Use fuzzy matching - check if any required sector matches any certificate sector
        is_met, matched_items, best_score = fuzzy_match_in_list(required, cert_sectors, threshold=0.3)
        matched = matched_items if is_met else []

        details['sectors'] = {
            'required': required,
            'matched': matched,
            'met': is_met,
            'match_score': round(best_score, 3) if is_met else 0.0
        }

        if is_met:
            filters_met += 1
            met_filters.append('sectors')
        else:
            unmet_filters.append('sectors')

    # Check Sub-Sectors (FUZZY MATCHING - JSONB array - ANY required sub-sector matches - OR logic)
    if filters.get('sub_sectors') and len(filters['sub_sectors']) > 0:
        required = filters['sub_sectors']
        cert_sub_sectors = certificate.sub_sectors or []
        
        # Use fuzzy matching - check if any required sub-sector matches any certificate sub-sector
        is_met, matched_items, best_score = fuzzy_match_in_list(required, cert_sub_sectors, threshold=0.3)
        matched = matched_items if is_met else []

        details['sub_sectors'] = {
            'required': required,
            'matched': matched,
            'met': is_met,
            'match_score': round(best_score, 3) if is_met else 0.0
        }

        if is_met:
            filters_met += 1
            met_filters.append('sub_sectors')
        else:
            unmet_filters.append('sub_sectors')

    # Check Funding Agency (FUZZY MATCHING)
    if filters.get('funding_agencies') and len(filters['funding_agencies']) > 0:
        required = filters['funding_agencies']
        # Use fuzzy matching across funding_agency and related fields
        is_met, match_score, match_details = fuzzy_match_multi_field(
            required, certificate, 'funding_agency',
            related_fields=['project_name', 'description'],
            threshold=0.3
        )
        matched = match_details.get('matched_texts', [None])[0] if is_met else None

        details['funding_agencies'] = {
            'required': required,
            'matched': matched,
            'met': is_met,
            'match_score': round(match_score, 3) if is_met else 0.0,
            'match_details': match_details
        }

        if is_met:
            filters_met += 1
            met_filters.append('funding_agencies')
        else:
            unmet_filters.append('funding_agencies')

    # Check Role/Lead/JV (FUZZY MATCHING)
    if filters.get('roles') and len(filters['roles']) > 0:
        required = filters['roles']
        # Use fuzzy matching for role field
        is_met, match_score, match_details = fuzzy_match_multi_field(
            required, certificate, 'role_lead_jv',
            related_fields=['project_name'],
            threshold=0.3
        )
        matched = match_details.get('matched_texts', [None])[0] if is_met else None

        details['roles'] = {
            'required': required,
            'matched': matched,
            'met': is_met,
            'match_score': round(match_score, 3) if is_met else 0.0,
            'match_details': match_details
        }

        if is_met:
            filters_met += 1
            met_filters.append('roles')
        else:
            unmet_filters.append('roles')

    # Check Duration (FUZZY MATCHING)
    if filters.get('durations') and len(filters['durations']) > 0:
        required = filters['durations']
        # Use fuzzy matching for duration field
        is_met, match_score, match_details = fuzzy_match_multi_field(
            required, certificate, 'duration',
            related_fields=['project_name'],
            threshold=0.3
        )
        matched = match_details.get('matched_texts', [None])[0] if is_met else None

        details['durations'] = {
            'required': required,
            'matched': matched,
            'met': is_met,
            'match_score': round(match_score, 3) if is_met else 0.0,
            'match_details': match_details
        }

        if is_met:
            filters_met += 1
            met_filters.append('durations')
        else:
            unmet_filters.append('durations')

    # Check Certificate Number (FUZZY MATCHING)
    if filters.get('certificate_numbers') and len(filters['certificate_numbers']) > 0:
        required = filters['certificate_numbers']
        # Use fuzzy matching for certificate_number field
        is_met, match_score, match_details = fuzzy_match_multi_field(
            required, certificate, 'certificate_number',
            related_fields=[],
            threshold=0.5  # Higher threshold for certificate numbers (more exact matching)
        )
        matched = match_details.get('matched_texts', [None])[0] if is_met else None

        details['certificate_numbers'] = {
            'required': required,
            'matched': matched,
            'met': is_met,
            'match_score': round(match_score, 3) if is_met else 0.0,
            'match_details': match_details
        }

        if is_met:
            filters_met += 1
            met_filters.append('certificate_numbers')
        else:
            unmet_filters.append('certificate_numbers')

    # Check Consultancy Fee Range (skip if min=0 even if max is set)
    if filters.get('consultancy_fee_range'):
        fee_range = filters['consultancy_fee_range']
        min_fee = fee_range.get('min')
        max_fee = fee_range.get('max')

        # STRICT: Only process if min > 0 (don't count if min=0 even if max is set)
        is_meaningful = False
        if min_fee is not None and min_fee > 0 and max_fee is not None and max_fee < 100000000:
            is_meaningful = True  # Both bounds are constrained
        elif min_fee is not None and min_fee > 0 and (max_fee is None or max_fee >= 100000000):
            is_meaningful = True  # Only min is constrained

        if is_meaningful:
            cert_fee = certificate.consultancy_fee_numeric

            is_met = False
            if cert_fee is not None:
                if min_fee is not None and max_fee is not None:
                    is_met = min_fee <= cert_fee <= max_fee
                elif min_fee is not None:
                    is_met = cert_fee >= min_fee
                elif max_fee is not None:
                    is_met = cert_fee <= max_fee

            details['consultancy_fee_range'] = {
                'required': {'min': min_fee, 'max': max_fee},
                'matched': cert_fee,
                'met': is_met
            }

            if is_met:
                filters_met += 1
                met_filters.append('consultancy_fee_range')
            else:
                unmet_filters.append('consultancy_fee_range')

    # Check Project Value Range (skip if min=0 even if max is set)
    if filters.get('project_value_range'):
        value_range = filters['project_value_range']
        min_value = value_range.get('min')
        max_value = value_range.get('max')

        # STRICT: Only process if min > 0 (don't count if min=0 even if max is set)
        is_meaningful = False
        if min_value is not None and min_value > 0 and max_value is not None and max_value < 1000000000:
            is_meaningful = True  # Both bounds are constrained
        elif min_value is not None and min_value > 0 and (max_value is None or max_value >= 1000000000):
            is_meaningful = True  # Only min is constrained

        if is_meaningful:
            cert_value = certificate.project_value

            is_met = False
            if cert_value is not None:
                if min_value is not None and max_value is not None:
                    is_met = min_value <= cert_value <= max_value
                elif min_value is not None:
                    is_met = cert_value >= min_value
                elif max_value is not None:
                    is_met = cert_value <= max_value

            details['project_value_range'] = {
                'required': {'min': min_value, 'max': max_value},
                'matched': cert_value,
                'met': is_met
            }

            if is_met:
                filters_met += 1
                met_filters.append('project_value_range')
            else:
                unmet_filters.append('project_value_range')

    # Check Completion Date Range
    if filters.get('completion_date_range'):
        date_range = filters['completion_date_range']
        start_date = date_range.get('start')
        end_date = date_range.get('end')
        cert_date = certificate.completion_date

        is_met = False
        if cert_date is not None:
            from datetime import datetime

            # Parse dates if they're strings
            if isinstance(start_date, str):
                start_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            if isinstance(end_date, str):
                end_date = datetime.fromisoformat(end_date.replace('Z', '+00:00'))

            if start_date and end_date:
                is_met = start_date <= cert_date <= end_date
            elif start_date:
                is_met = cert_date >= start_date
            elif end_date:
                is_met = cert_date <= end_date

        details['completion_date_range'] = {
            'required': {'start': start_date, 'end': end_date},
            'matched': cert_date,
            'met': is_met
        }

        if is_met:
            filters_met += 1
            met_filters.append('completion_date_range')
        else:
            unmet_filters.append('completion_date_range')

    # Calculate compliance percentage
    compliance_percentage = (filters_met / total_filters * 100) if total_filters > 0 else 0

    return {
        'total_filters': total_filters,
        'filters_met': filters_met,
        'compliance_percentage': round(compliance_percentage, 2),
        'met_filters': met_filters,
        'unmet_filters': unmet_filters,
        'details': details
    }


def get_compliance_levels(total_filters: int) -> list:
    """
    Get dynamic compliance levels based on total filters applied.

    Rules:
    - 1-2 filters: Show all levels
    - 3 filters: Show 3/3, 2/3, 1/3
    - 4+ filters: Show top 4 levels
    """
    if total_filters == 0:
        return []
    elif total_filters <= 2:
        return list(range(total_filters, 0, -1))
    elif total_filters == 3:
        return [3, 2, 1]
    else:  # 4 or more
        return list(range(total_filters, max(0, total_filters - 4), -1))


@router.post("/api/certificates/manual-clause/search")
@require_company_details
async def manual_clause_search(
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Manual Clause Search with Compliance Scoring.

    Searches certificates based on multiple filter criteria and calculates
    compliance scores showing how many filters each certificate meets.

    Returns certificates sorted by compliance level (highest first) with
    detailed compliance information.
    """
    current_user = get_current_user(request, db)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        # Parse request body
        body = await request.json()
        filters = body.get('filters', {})

        # 🔍 DEBUG: Log the exact filters received
        logger.info("🔍 RAW FILTERS RECEIVED FROM FRONTEND:")
        logger.info(f"   Full filters dict: {filters}")
        logger.info(f"   Filter keys: {list(filters.keys())}")
        for key, value in filters.items():
            logger.info(f"   {key}: {value} (type: {type(value).__name__})")

        # Reset the logging flag for this new request
        if hasattr(calculate_compliance, '_logged_filters'):
            delattr(calculate_compliance, '_logged_filters')

        # Count active filters
        total_filters = count_active_filters(filters)
        logger.info(f"🔢 count_active_filters() returned: {total_filters}")

        if total_filters == 0:
            return {
                "success": False,
                "error": "At least one filter must be applied",
                "results": [],
                "compliance_summary": {
                    "total_filters_applied": 0,
                    "compliance_levels": []
                }
            }

        logger.info("🚀 " + "=" * 80)
        logger.info(f"🔍 MANUAL CLAUSE SEARCH REQUEST - User: {current_user.email}")
        logger.info(f"📊 Total Filters Applied: {total_filters}")
        logger.info("=" * 80)

        # Fetch ALL completed certificates for the user
        certificates = db.query(CertificateDB).filter(
            CertificateDB.user_id == current_user.id,
            CertificateDB.processing_status == "completed"
        ).all()

        logger.info(f"📊 Found {len(certificates)} total certificates to evaluate")

        # Calculate compliance for each certificate
        results = []
        compliance_counts = {}  # Track counts for each compliance level

        for cert in certificates:
            compliance = calculate_compliance(cert, filters)

            # Only include certificates with at least 1 filter met (exclude 0 criteria met)
            if compliance['filters_met'] > 0:
                # Track compliance level counts (excluding level 0)
                level = compliance['filters_met']
                compliance_counts[level] = compliance_counts.get(level, 0) + 1

                results.append({
                    'certificate': {
                        'id': cert.id,
                        'project_name': cert.project_name,
                        'client_name': cert.client_name,
                        'location': cert.location,
                        'services_rendered': cert.services_rendered,
                        'sectors': cert.sectors,
                        'sub_sectors': cert.sub_sectors,
                        'funding_agency': cert.funding_agency,
                        'role_lead_jv': cert.role_lead_jv,
                        'duration': cert.duration,
                        'certificate_number': cert.certificate_number,
                        'consultancy_fee_numeric': cert.consultancy_fee_numeric,
                        'consultancy_fee_inr': cert.consultancy_fee_inr,
                        'project_value': cert.project_value,
                        'project_value_inr': cert.project_value_inr,
                        'completion_date': cert.completion_date.isoformat() if cert.completion_date else None,
                        'created_at': cert.created_at.isoformat() if cert.created_at else None,
                    },
                    'compliance': compliance
                })

        # Sort results by compliance (highest first), then by created_at (newest first)
        results.sort(key=lambda x: (-x['compliance']['filters_met'], x['certificate']['created_at']), reverse=False)

        # Build compliance summary
        compliance_levels = get_compliance_levels(total_filters)
        compliance_summary = {
            'total_filters_applied': total_filters,
            'compliance_levels': [
                {
                    'level': f"{level}/{total_filters}",
                    'count': compliance_counts.get(level, 0)
                }
                for level in compliance_levels
            ]
        }

        # Log results summary
        logger.info("=" * 100)
        logger.info("✅ SEARCH COMPLETE - RESULTS SUMMARY")
        logger.info("=" * 100)
        logger.info(f"📁 Total Certificates Evaluated: {len(certificates)}")
        logger.info(f"✨ Matching Certificates Found: {len(results)}")

        if compliance_counts:
            logger.info("📊 Compliance Distribution:")
            for level in sorted(compliance_counts.keys(), reverse=True):
                logger.info(f"   {level}/{total_filters} filters met: {compliance_counts[level]} certificate(s)")

        logger.info("=" * 100)

        return {
            "success": True,
            "results": results,
            "compliance_summary": compliance_summary
        }

    except Exception as e:
        logger.error(f"❌ Error in manual clause search: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


def extract_json_from_response(response: str) -> dict:
    """
    Extract JSON from AI response, handling markdown code blocks and other formatting.

    Args:
        response: Raw response string from AI

    Returns:
        Parsed JSON as dictionary

    Raises:
        json.JSONDecodeError: If no valid JSON found
    """
    if not response:
        raise json.JSONDecodeError("Empty response", "", 0)

    # Strip whitespace
    response = response.strip()

    # Try direct JSON parse first
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        # Store original error for later
        original_error = e

    # Try to extract JSON from markdown code blocks
    # Pattern 1: ```json ... ```
    if '```json' in response:
        try:
            start = response.find('```json') + 7
            end = response.find('```', start)
            if end != -1:
                json_str = response[start:end].strip()
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # Pattern 2: ``` ... ```
    if '```' in response:
        try:
            start = response.find('```') + 3
            end = response.find('```', start)
            if end != -1:
                json_str = response[start:end].strip()
                # Remove language identifier if present (e.g., "json\n")
                if json_str.startswith('json\n'):
                    json_str = json_str[5:]
                elif json_str.startswith('json '):
                    json_str = json_str[5:]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # Try to find JSON object boundaries
    # Look for first { and last }
    start = response.find('{')
    end = response.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            json_str = response[start:end+1]
            parsed = json.loads(json_str)
            return parsed
        except json.JSONDecodeError:
            # Try to fix truncated JSON by adding closing brackets
            try:
                # Count opening and closing braces/brackets
                open_braces = json_str.count('{')
                close_braces = json_str.count('}')
                open_brackets = json_str.count('[')
                close_brackets = json_str.count(']')

                # Add missing closing characters
                fixed_str = json_str
                if close_brackets < open_brackets:
                    fixed_str += ']' * (open_brackets - close_brackets)
                if close_braces < open_braces:
                    fixed_str += '}' * (open_braces - close_braces)

                return json.loads(fixed_str)
            except json.JSONDecodeError:
                pass

    # If all else fails, raise descriptive error
    logger.error(f"Failed to parse JSON. Response length: {len(response)}")
    logger.error(f"First 500 chars: {response[:500]}")
    logger.error(f"Last 200 chars: {response[-200:]}")
    raise json.JSONDecodeError(f"No valid JSON found in response (length: {len(response)})", response[:100], 0)


@router.post("/api/certificates/ai-extract-clauses")
@require_company_details
async def ai_extract_clauses(
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Extract tender clauses from text and/or images using GPT-4o.

    Input: { "text": "...", "images": ["base64_image1", ...] }
    Output: { "clauses": {...} }
    """
    current_user = get_current_user(request, db)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        data = await request.json()
        text_input = (data.get('text') or '').strip()
        images = data.get('images') or []

        if not text_input and not images:
            raise HTTPException(status_code=400, detail="Please provide text or images")

        # Get OpenAI API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")

        ai_client = OpenAIWrapper(api_key)

        # Clause extraction prompt
        system_prompt = """You are an expert at analyzing tender documents and extracting key requirements.

Analyze the provided text/images and extract the following information.

CRITICAL: Your response must be ONLY a valid JSON object, with no markdown formatting, no code blocks, no explanatory text before or after. Start with { and end with }.

Required JSON structure:
{
  "location": "Extracted location/city/state (single string)",
  "sectors": ["Sector 1", "Sector 2"],
  "sub_sectors": ["Sub-sector 1"],
  "services": ["Service 1", "Service 2"],
  "client_type": "Type of client organization",
  "funding_agency": "Funding agency if mentioned",
  "project_value_min": number or null,
  "project_value_max": number or null,
  "consultancy_fee_min": number or null,
  "consultancy_fee_max": number or null,
  "duration": "Project duration as mentioned",
  "role_requirement": "Lead/JV/Consortium/etc",
  "completion_date_start": "YYYY-MM-DD or null",
  "completion_date_end": "YYYY-MM-DD or null"
}

Extract only information explicitly stated. Use null for missing fields.
DO NOT wrap your response in ```json or ``` blocks. Return ONLY the raw JSON object."""

        # Build messages for GPT-4o
        messages = [{"role": "system", "content": system_prompt}]

        if images and len(images) > 0:
            # GPT-4o with vision
            content_parts = []

            if text_input:
                content_parts.append({"type": "text", "text": f"Text input: {text_input}"})
            else:
                content_parts.append({"type": "text", "text": "Analyze these tender document images and extract the clause information."})

            # Add images (limit to first 5 for cost control)
            for img_base64 in images[:5]:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}",
                        "detail": "high"
                    }
                })

            messages.append({"role": "user", "content": content_parts})
        else:
            # Text only
            messages.append({"role": "user", "content": text_input})

        logger.info(f"🤖 Extracting clauses using GPT-4o for user {current_user.email}")

        # Call GPT-4o
        response = ai_client.chat_completion(
            messages=messages,
            model="gpt-4o",
            max_completion_tokens=2000,
            temperature=0.1,
            timeout=60.0
        )

        logger.info(f"Raw GPT-4o response (first 200 chars): {response[:200]}")

        # Parse JSON response (handle markdown code blocks)
        clauses = extract_json_from_response(response)

        logger.info(f"✅ Clause extraction complete: {list(clauses.keys())}")

        return {"success": True, "clauses": clauses}

    except json.JSONDecodeError as e:
        logger.error(f"❌ Failed to parse AI response as JSON: {e}")
        logger.error(f"Response content: {response[:500] if 'response' in locals() else 'N/A'}")
        raise HTTPException(status_code=500, detail="AI returned invalid JSON response")

    except OpenAIServiceError as e:
        logger.error(f"❌ OpenAI API error: {e}")
        raise HTTPException(status_code=500, detail=f"AI service error: {str(e)}")

    except Exception as e:
        logger.error(f"❌ Error in clause extraction: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Extraction error: {str(e)}")


@router.post("/api/certificates/ai-extract-conditions")
@require_company_details
async def ai_extract_conditions(
    request: Request,
    db: Session = Depends(get_db)
):
    """
    NEW AI EXTRACTION: Extract tender REQUIREMENTS/CONDITIONS directly as filter values.
    This replaces the 2-stage pipeline with a single GPT-4o call that extracts
    what the tender REQUIRES companies to have, not what exists in dropdowns.

    Input: { "text": "...", "images": ["base64_image1", ...] }
    Output: { "filters": {...}, "raw_extraction": {...} }
    """
    current_user = get_current_user(request, db)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        data = await request.json()
        text_input = (data.get('text') or '').strip()
        images = data.get('images') or []

        if not text_input and not images:
            raise HTTPException(status_code=400, detail="Please provide text or images")

        # Get OpenAI API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")

        ai_client = OpenAIWrapper(api_key)

        # ENHANCED CONDITION-FOCUSED PROMPT
        system_prompt = """You are analyzing a TENDER DOCUMENT to extract the ELIGIBILITY CRITERIA and REQUIREMENTS that companies must meet to qualify.

Your goal: Extract CONDITIONS that applicants must satisfy, not general project descriptions.

Focus on these requirement types:

1. **Location Requirements**: Where must the company have prior experience? Extract ALL mentioned locations (cities, states, regions, countries).

2. **Sector/Domain Experience**: What sectors/industries? (Water Supply, Healthcare, Transport, Urban Development, Energy, Education, etc.)

3. **Services Required**: What type of work must they have done? (Feasibility Study, DPR Preparation, PMC, Project Management, Detailed Engineering, Supervision, etc.)

4. **Client Type**: Who should prior clients be? (Government, PSU, Private Sector, Municipal Corporation, International Organization, State Government, Central Government, etc.)

5. **Funding Agency**: Any specific funding body experience required? (World Bank, ADB, JICA, DFID, KfW, etc.)

6. **Financial Criteria**:
   - Minimum consultancy fee from past similar projects (in INR)
   - Minimum project value handled (in INR)

7. **Duration**: Required project duration experience (e.g., "12 months", "18 months", "2 years")

8. **Role**: What role experience needed? (Lead Consultant, JV Partner, Consortium Member, Solo Consultant)

9. **Completion Date**: When should past projects have been completed?
   - After date: Projects completed after this date qualify
   - Before date: Projects completed before this date qualify

10. **Certificate Date**: How recent should experience certificates be?
    - After date: Certificates issued after this date are valid

EXTRACTION RULES:
- Extract MULTIPLE values per field if tender mentions multiple options
- Use EXACT TERMS from tender text (don't paraphrase or simplify)
- For "similar" or "related" work, extract the specific examples given
- For "OR" conditions ("Water Supply OR Sanitation"), extract both as separate items
- For ranges: extract min/max values; use null if unbounded
- For dates: use "YYYY-MM-DD" format or null if not specified
- If tender says "preference for", still extract it as a condition

CRITICAL OUTPUT RULES:
- Return ONLY valid JSON, no markdown, no code blocks, no explanation
- ONLY include fields that are EXPLICITLY mentioned in the tender
- DO NOT include fields with empty arrays [] or null values
- If a field is not mentioned in the tender, OMIT it entirely from the JSON
- For ranges: only include if at least one bound (min or max) is specified
- For dates: only include if at least one date (after or before) is mentioned

Example 1: If tender says "Experience in Water Supply projects in Delhi or NCR region with Government clients, minimum project value Rs. 10 Crores, completed in last 5 years", extract:
{
  "location": ["Delhi", "NCR"],
  "sectors": ["Water Supply"],
  "client_type": ["Government"],
  "project_value_range": {"min": 100000000, "max": null},
  "completion_date": {"after": "2020-01-01", "before": null}
}

Example 2: If tender says "Looking for consultants with PMC experience in Urban Development sector":
{
  "services": ["PMC"],
  "sectors": ["Urban Development"]
}

Example 3: If tender only says "Projects in Maharashtra":
{
  "location": ["Maharashtra"]
}

DO NOT wrap response in ```json blocks. Return raw JSON only."""

        # Build messages for GPT-4o
        messages = [{"role": "system", "content": system_prompt}]

        if images and len(images) > 0:
            # GPT-4o with vision
            content_parts = []

            if text_input:
                content_parts.append({
                    "type": "text",
                    "text": f"Tender Document Text:\n\n{text_input}\n\nExtract the eligibility requirements and conditions from this tender."
                })
            else:
                content_parts.append({
                    "type": "text",
                    "text": "Analyze these tender document images and extract the eligibility requirements and conditions that companies must meet."
                })

            # Add images (limit to first 5 for cost control)
            for img_base64 in images[:5]:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}",
                        "detail": "high"
                    }
                })

            messages.append({"role": "user", "content": content_parts})
        else:
            # Text only
            messages.append({
                "role": "user",
                "content": f"Tender Document Text:\n\n{text_input}\n\nExtract the eligibility requirements and conditions from this tender."
            })

        logger.info(f"🤖 Extracting tender CONDITIONS using GPT-4o for user {current_user.email}")

        # Call GPT-4o
        response = ai_client.chat_completion(
            messages=messages,
            model="gpt-4o",
            max_completion_tokens=2500,
            temperature=0.1,
            timeout=60.0
        )

        logger.info(f"Raw GPT-4o response (first 300 chars): {response[:300]}")

        # Parse JSON response (handle markdown code blocks)
        conditions = extract_json_from_response(response)

        # Log extracted conditions
        extracted_count = sum(1 for k, v in conditions.items() if v and (isinstance(v, list) and len(v) > 0) or (isinstance(v, dict) and any(v.values())))
        logger.info(f"✅ Condition extraction complete: {extracted_count} filter types extracted")
        logger.info(f"Extracted filter keys: {list(conditions.keys())}")

        return {
            "success": True,
            "filters": conditions,
            "raw_extraction": conditions  # Keep for debugging
        }

    except json.JSONDecodeError as e:
        logger.error(f"❌ Failed to parse AI response as JSON: {e}")
        logger.error(f"Response content: {response[:500] if 'response' in locals() else 'N/A'}")
        raise HTTPException(status_code=500, detail="AI returned invalid JSON response")

    except OpenAIServiceError as e:
        logger.error(f"❌ OpenAI API error: {e}")
        raise HTTPException(status_code=500, detail=f"AI service error: {str(e)}")

    except Exception as e:
        logger.error(f"❌ Error in condition extraction: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Extraction error: {str(e)}")


@router.post("/api/certificates/extract-keywords")
@require_company_details
async def extract_keywords(
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Extract keywords from clause text/images by using AI extraction and converting
    filter values into a flat list of searchable keywords.
    
    Input: { "text": str, "images": List[str] }
    Output: { "success": bool, "keywords": List[str], "source_filters": Dict }
    """
    current_user = get_current_user(request, db)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        data = await request.json()
        text_input = (data.get('text') or '').strip()
        images = data.get('images') or []
        
        if not text_input and not images:
            raise HTTPException(status_code=400, detail="Please provide text or images")
        
        # Use existing ai_extract_conditions endpoint internally
        # Create a mock request object to call the function
        from fastapi import Request as FastAPIRequest
        from starlette.datastructures import Headers
        
        # Call the extraction function directly
        extraction_result = await ai_extract_conditions(request, db)
        filters = extraction_result.get('filters', {})
        
        # Convert filter values to flat keyword list
        keywords = []
        
        # Extract from location array
        if 'location' in filters and isinstance(filters['location'], list):
            keywords.extend([str(loc).strip() for loc in filters['location'] if loc])
        
        # Extract from sectors array
        if 'sectors' in filters and isinstance(filters['sectors'], list):
            keywords.extend([str(sector).strip() for sector in filters['sectors'] if sector])
        
        # Extract from sub_sectors array
        if 'sub_sectors' in filters and isinstance(filters['sub_sectors'], list):
            keywords.extend([str(sub).strip() for sub in filters['sub_sectors'] if sub])
        
        # Extract from services array
        if 'services' in filters and isinstance(filters['services'], list):
            keywords.extend([str(service).strip() for service in filters['services'] if service])
        
        # Extract from client_type array
        if 'client_type' in filters and isinstance(filters['client_type'], list):
            keywords.extend([str(client).strip() for client in filters['client_type'] if client])
        
        # Extract from funding_agency array
        if 'funding_agency' in filters and isinstance(filters['funding_agency'], list):
            keywords.extend([str(agency).strip() for agency in filters['funding_agency'] if agency])
        
        # Extract from roles array
        if 'roles' in filters and isinstance(filters['roles'], list):
            keywords.extend([str(role).strip() for role in filters['roles'] if role])
        
        # Extract from durations array
        if 'durations' in filters and isinstance(filters['durations'], list):
            keywords.extend([str(duration).strip() for duration in filters['durations'] if duration])
        
        # Extract numeric values with units from ranges
        if 'consultancy_fee_range' in filters and isinstance(filters['consultancy_fee_range'], dict):
            fee_range = filters['consultancy_fee_range']
            if 'min' in fee_range and fee_range['min']:
                # Convert to crores/lakhs format
                min_val = fee_range['min']
                if min_val >= 10000000:  # >= 1 crore
                    keywords.append(f"{min_val / 10000000} crores")
                    keywords.append(f"{int(min_val / 10000000)} crores")
                elif min_val >= 100000:  # >= 1 lakh
                    keywords.append(f"{min_val / 100000} lakhs")
                    keywords.append(f"{int(min_val / 100000)} lakhs")
                keywords.append(str(min_val))
            if 'max' in fee_range and fee_range['max']:
                max_val = fee_range['max']
                if max_val >= 10000000:
                    keywords.append(f"{max_val / 10000000} crores")
                    keywords.append(f"{int(max_val / 10000000)} crores")
                elif max_val >= 100000:
                    keywords.append(f"{max_val / 100000} lakhs")
                    keywords.append(f"{int(max_val / 100000)} lakhs")
                keywords.append(str(max_val))
        
        if 'project_value_range' in filters and isinstance(filters['project_value_range'], dict):
            value_range = filters['project_value_range']
            if 'min' in value_range and value_range['min']:
                min_val = value_range['min']
                if min_val >= 10000000:
                    keywords.append(f"{min_val / 10000000} crores")
                    keywords.append(f"{int(min_val / 10000000)} crores")
                elif min_val >= 100000:
                    keywords.append(f"{min_val / 100000} lakhs")
                    keywords.append(f"{int(min_val / 100000)} lakhs")
                keywords.append(str(min_val))
            if 'max' in value_range and value_range['max']:
                max_val = value_range['max']
                if max_val >= 10000000:
                    keywords.append(f"{max_val / 10000000} crores")
                    keywords.append(f"{int(max_val / 10000000)} crores")
                elif max_val >= 100000:
                    keywords.append(f"{max_val / 100000} lakhs")
                    keywords.append(f"{int(max_val / 100000)} lakhs")
                keywords.append(str(max_val))
        
        # Extract date information as keywords
        if 'completion_date' in filters and isinstance(filters['completion_date'], dict):
            date_range = filters['completion_date']
            if 'after' in date_range and date_range['after']:
                keywords.append(f"after {date_range['after']}")
            if 'before' in date_range and date_range['before']:
                keywords.append(f"before {date_range['before']}")
        
        # Remove duplicates and empty strings
        keywords = list(set([kw for kw in keywords if kw and kw.strip()]))
        
        # Limit keywords to prevent performance issues
        MAX_KEYWORDS = 50
        if len(keywords) > MAX_KEYWORDS:
            logger.warning(f"⚠️ Limiting keywords from {len(keywords)} to {MAX_KEYWORDS}")
            keywords = keywords[:MAX_KEYWORDS]
        
        if len(keywords) == 0:
            logger.warning("⚠️ No keywords extracted from filters")
            return {
                "success": True,
                "keywords": [],
                "source_filters": filters,
                "message": "No keywords could be extracted from the clause"
            }
        
        logger.info(f"✅ Extracted {len(keywords)} keywords from filters for user {current_user.email}")
        logger.info(f"Keywords: {keywords[:10]}...")  # Log first 10
        
        return {
            "success": True,
            "keywords": keywords,
            "source_filters": filters
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in keyword extraction: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Keyword extraction error: {str(e)}")


@router.post("/api/certificates/expand-keywords")
@require_company_details
async def expand_keywords(
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Expand keywords using AI to generate variations, synonyms, and alternative phrasings.
    Makes individual AI calls for each keyword to generate variations.
    
    Input: { "keywords": List[str] }
    Output: { "success": bool, "expanded": { "keyword": ["variant1", "variant2", ...] } }
    """
    current_user = get_current_user(request, db)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        data = await request.json()
        keywords = data.get('keywords', [])
        
        if not keywords or not isinstance(keywords, list):
            raise HTTPException(status_code=400, detail="Please provide a list of keywords")
        
        if len(keywords) == 0:
            return {
                "success": True,
                "expanded": {}
            }
        
        # Limit keywords to prevent API rate limiting
        MAX_KEYWORDS = 30
        if len(keywords) > MAX_KEYWORDS:
            logger.warning(f"⚠️ Limiting keyword expansion from {len(keywords)} to {MAX_KEYWORDS}")
            keywords = keywords[:MAX_KEYWORDS]
        
        # Get OpenAI API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")
        
        ai_client = OpenAIWrapper(api_key)
        
        expanded_results = {}
        
        # Expand each keyword individually
        for keyword in keywords:
            if not keyword or not str(keyword).strip():
                continue
            
            keyword_str = str(keyword).strip()
            
            # Create expansion prompt
            expansion_prompt = f"""Generate all common ways to write or express this keyword/phrase. Include:
1. Numeric variations (e.g., "100" -> "hundred", "one hundred")
2. Unit variations (e.g., "sq km" -> "square km", "square kilometers", "sq kilometers")
3. Abbreviations and full forms (e.g., "NCR" -> "National Capital Region")
4. Synonyms and alternative phrasings
5. Common misspellings or variations
6. Different word orders if applicable

Keyword: "{keyword_str}"

Return ONLY a JSON array of variations, no explanation, no markdown. Example format:
["variant1", "variant2", "variant3"]

If the keyword is already very specific or technical, return variations that maintain the same meaning."""
            
            try:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant that generates keyword variations for search purposes. Always return valid JSON arrays only."},
                    {"role": "user", "content": expansion_prompt}
                ]
                
                response = ai_client.chat_completion(
                    messages=messages,
                    model="gpt-4o",
                    max_completion_tokens=500,
                    temperature=0.3,
                    timeout=30.0
                )
                
                # Parse JSON array from response
                # Try to extract JSON array
                import re
                json_match = re.search(r'\[.*?\]', response, re.DOTALL)
                if json_match:
                    variants = json.loads(json_match.group())
                    # Ensure keyword itself is included
                    if keyword_str not in variants:
                        variants.insert(0, keyword_str)
                    expanded_results[keyword_str] = variants
                else:
                    # Fallback: just use the keyword itself
                    expanded_results[keyword_str] = [keyword_str]
                
                logger.info(f"✅ Expanded '{keyword_str}' to {len(expanded_results[keyword_str])} variants")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to expand keyword '{keyword_str}': {e}. Using keyword as-is.")
                expanded_results[keyword_str] = [keyword_str]
        
        logger.info(f"✅ Expanded {len(expanded_results)} keywords for user {current_user.email}")
        
        return {
            "success": True,
            "expanded": expanded_results
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in keyword expansion: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Keyword expansion error: {str(e)}")


def search_certificate_by_keyword(keyword: str, variants: List[str], user_id: str, db: Session) -> List[str]:
    """
    Search certificates for a keyword and its variants across multiple fields.
    
    Args:
        keyword: The original keyword
        variants: List of expanded variants of the keyword
        user_id: User ID to filter certificates
        db: Database session
    
    Returns:
        List of certificate IDs that match the keyword or any of its variants
    """
    # Combine keyword with variants for search
    search_terms = [keyword] + variants
    search_terms = [term.lower().strip() for term in search_terms if term and term.strip()]
    
    if not search_terms:
        return []
    
    # Build OR conditions for each search term
    conditions = []
    for term in search_terms:
        search_pattern = f"%{term}%"
        conditions.append(
            or_(
                CertificateDB.project_name.ilike(search_pattern),
                CertificateDB.client_name.ilike(search_pattern),
                CertificateDB.location.ilike(search_pattern),
                CertificateDB.extracted_text.ilike(search_pattern),
                CertificateDB.funding_agency.ilike(search_pattern),
                CertificateDB.role_lead_jv.ilike(search_pattern),
                CertificateDB.duration.ilike(search_pattern),
                CertificateDB.certificate_number.ilike(search_pattern),
                # Search in JSONB arrays
                cast(CertificateDB.services_rendered, String).ilike(search_pattern),
                cast(CertificateDB.sectors, String).ilike(search_pattern),
                cast(CertificateDB.sub_sectors, String).ilike(search_pattern)
            )
        )
    
    # Combine all conditions with OR
    combined_condition = or_(*conditions)
    
    # Query certificates
    matching_certs = db.query(CertificateDB.id).filter(
        CertificateDB.user_id == user_id,
        CertificateDB.processing_status == "completed",
        combined_condition
    ).all()
    
    return [str(cert.id) for cert in matching_certs]


def calculate_keyword_matches(certificates: List[CertificateDB], keyword_results: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    """
    Calculate how many keywords each certificate matches.
    
    Args:
        certificates: List of certificate objects
        keyword_results: Dict mapping keyword to list of certificate IDs that match it
    
    Returns:
        List of dicts with certificate data and match information, sorted by match count
    """
    # Create a mapping of certificate ID to certificate object
    cert_dict = {str(cert.id): cert for cert in certificates}
    
    # Count matches for each certificate
    match_results = []
    for cert_id, cert in cert_dict.items():
        matched_keywords = []
        match_count = 0
        
        # Check each keyword
        for keyword, matching_cert_ids in keyword_results.items():
            if cert_id in matching_cert_ids:
                matched_keywords.append(keyword)
                match_count += 1
        
        if match_count > 0:  # Only include certificates with at least one match
            match_results.append({
                'certificate': cert,
                'matched_keywords': matched_keywords,
                'match_count': match_count,
                'total_keywords': len(keyword_results)
            })
    
    # Sort by match count (highest first), then by created_at (newest first)
    match_results.sort(
        key=lambda x: (-x['match_count'], x['certificate'].created_at if x['certificate'].created_at else datetime.min),
        reverse=False
    )
    
    return match_results


@router.post("/api/certificates/search-by-keywords")
@require_company_details
async def search_by_keywords(
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Search certificates using keywords and their expanded variants.
    Returns certificates sorted by how many keywords they match (weighted intersection).
    
    Input: {
        "keywords": List[str],
        "expanded_variants": { "keyword": ["variant1", "variant2", ...] }
    }
    Output: {
        "success": bool,
        "results": [...],
        "summary": {...}
    }
    """
    current_user = get_current_user(request, db)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        data = await request.json()
        keywords = data.get('keywords', [])
        expanded_variants = data.get('expanded_variants', {})
        
        if not keywords or not isinstance(keywords, list):
            raise HTTPException(status_code=400, detail="Please provide a list of keywords")
        
        if len(keywords) == 0:
            return {
                "success": True,
                "results": [],
                "summary": {
                    "total_keywords": 0,
                    "total_certificates_found": 0
                }
            }
        
        logger.info(f"🔍 Searching certificates for {len(keywords)} keywords for user {current_user.email}")
        
        # Store search results in memory: { keyword: [certificate_ids] }
        keyword_results = {}
        
        # Search for each keyword
        for keyword in keywords:
            if not keyword or not str(keyword).strip():
                continue
            
            keyword_str = str(keyword).strip()
            variants = expanded_variants.get(keyword_str, [])
            
            # Search certificates for this keyword and its variants
            matching_cert_ids = search_certificate_by_keyword(
                keyword_str,
                variants,
                current_user.id,
                db
            )
            
            keyword_results[keyword_str] = matching_cert_ids
            logger.info(f"  Keyword '{keyword_str}': {len(matching_cert_ids)} certificates matched")
        
        # Get all unique certificate IDs
        all_cert_ids = set()
        for cert_ids in keyword_results.values():
            all_cert_ids.update(cert_ids)
        
        if not all_cert_ids:
            logger.info("❌ No certificates matched any keywords")
            return {
                "success": True,
                "results": [],
                "summary": {
                    "total_keywords": len(keywords),
                    "total_certificates_found": 0
                }
            }
        
        # Fetch all matching certificates
        certificates = db.query(CertificateDB).filter(
            CertificateDB.id.in_(all_cert_ids),
            CertificateDB.user_id == current_user.id,
            CertificateDB.processing_status == "completed"
        ).all()
        
        # Calculate keyword matches
        match_results = calculate_keyword_matches(certificates, keyword_results)
        
        # Format results for response
        formatted_results = []
        for result in match_results:
            cert = result['certificate']
            formatted_results.append({
                'certificate': {
                    'id': cert.id,
                    'project_name': cert.project_name,
                    'client_name': cert.client_name,
                    'location': cert.location,
                    'services_rendered': cert.services_rendered,
                    'sectors': cert.sectors,
                    'sub_sectors': cert.sub_sectors,
                    'funding_agency': cert.funding_agency,
                    'role_lead_jv': cert.role_lead_jv,
                    'duration': cert.duration,
                    'certificate_number': cert.certificate_number,
                    'consultancy_fee_numeric': cert.consultancy_fee_numeric,
                    'consultancy_fee_inr': cert.consultancy_fee_inr,
                    'project_value': cert.project_value,
                    'project_value_inr': cert.project_value_inr,
                    'completion_date': cert.completion_date.isoformat() if cert.completion_date else None,
                    'created_at': cert.created_at.isoformat() if cert.created_at else None,
                },
                'matched_keywords': result['matched_keywords'],
                'match_count': result['match_count'],
                'total_keywords': result['total_keywords']
            })
        
        logger.info(f"✅ Search complete: {len(formatted_results)} certificates found")
        
        return {
            "success": True,
            "results": formatted_results,
            "summary": {
                "total_keywords": len(keywords),
                "total_certificates_found": len(formatted_results)
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in keyword-based search: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@router.post("/api/certificates/ai-match-filters")
@require_company_details
async def ai_match_filters(
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Match extracted clauses to filter options using 12 parallel GPT-3.5-turbo calls.

    Input: { "clauses": {...}, "filter_options": {...} }
    Output: { "matched_filters": {...}, "confidence_scores": {...} }
    """
    current_user = get_current_user(request, db)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        data = await request.json()
        clauses = data.get('clauses', {})
        filter_options = data.get('filter_options', {})

        if not clauses:
            raise HTTPException(status_code=400, detail="No clauses provided")

        # Get OpenAI API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")

        ai_client = OpenAIWrapper(api_key)

        logger.info(f"🤖 Matching filters using 12 parallel GPT calls for user {current_user.email}")

        # Define matching tasks for each filter type
        async def match_single_filter(filter_type: str, extracted_value, available_options: List):
            """Match a single filter type using GPT-3.5-turbo with semantic similarity."""
            if not extracted_value or not available_options:
                return None, 0.0

            # Handle different input types
            if isinstance(extracted_value, list):
                extracted_str = ", ".join(str(v) for v in extracted_value)
            else:
                extracted_str = str(extracted_value)

            # Create focused matching prompt for multiple matches
            prompt = f"""Available {filter_type} options: {json.dumps(available_options[:30])}
{f"...and {len(available_options)-30} more options" if len(available_options) > 30 else ""}

Extracted: {extracted_str}

Find the TOP 3-4 MOST RELEVANT matches using semantic similarity. Rules:
1. Return 1-4 matches, sorted by relevance (most relevant first)
2. Each option ONCE only (no duplicates)
3. Use semantic understanding (e.g., "Water Supply" matches both "Water Supply" and "Water & Sanitation")
4. Return exact values from available options list only
5. If only 1 good match, return 1. If 2-4 good matches, return all
6. Empty array if no semantic match

Return ONLY this JSON (no other text):
{{"matched": ["MostRelevant", "AlsoRelevant", "SomewhatRelevant"], "confidence": 0.85}}

Confidence: 1.0 = exact match, 0.8+ = very similar, 0.6+ = related, <0.6 = weak"""

            try:
                response = ai_client.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    model="gpt-3.5-turbo",
                    temperature=0.1,
                    max_completion_tokens=300,
                    timeout=20.0
                )

                logger.info(f"Filter {filter_type} raw response: {response[:300]}")

                # Parse JSON response (handle markdown code blocks)
                result = extract_json_from_response(response)
                matched = result.get('matched', [])
                confidence = result.get('confidence', 0.0)

                # Deduplicate matches (in case GPT returns duplicates)
                matched = list(dict.fromkeys(matched))  # Preserves order, removes duplicates

                # Filter out any matches not in available options (safety check)
                validated_matches = [m for m in matched if m in available_options]

                # Cap at top 4 matches (most relevant first, as GPT sorted them)
                validated_matches = validated_matches[:4]

                logger.info(f"Filter {filter_type} matched: {validated_matches} (confidence: {confidence})")

                return validated_matches, confidence

            except Exception as e:
                logger.error(f"❌ Error matching {filter_type}: {e}")
                if 'response' in locals():
                    logger.error(f"Full response: {response}")
                return None, 0.0

        # Define 12 matching tasks
        tasks = []

        # 1. Clients (if client_type extracted)
        tasks.append(
            ('clients', match_single_filter(
                'clients',
                clauses.get('client_type'),
                filter_options.get('clients', [])
            ))
        )

        # 2. Locations
        tasks.append(
            ('locations', match_single_filter(
                'locations',
                clauses.get('location'),
                filter_options.get('locations', [])
            ))
        )

        # 3. Services
        tasks.append(
            ('services', match_single_filter(
                'services',
                clauses.get('services'),
                filter_options.get('services', [])
            ))
        )

        # 4. Sectors
        tasks.append(
            ('sectors', match_single_filter(
                'sectors',
                clauses.get('sectors'),
                filter_options.get('sectors', [])
            ))
        )

        # 5. Sub-sectors
        tasks.append(
            ('sub_sectors', match_single_filter(
                'sub_sectors',
                clauses.get('sub_sectors'),
                filter_options.get('sub_sectors', [])
            ))
        )

        # 6. Funding agencies
        tasks.append(
            ('funding_agencies', match_single_filter(
                'funding_agencies',
                clauses.get('funding_agency'),
                filter_options.get('funding_agencies', [])
            ))
        )

        # 7. Roles
        tasks.append(
            ('roles', match_single_filter(
                'roles',
                clauses.get('role_requirement'),
                filter_options.get('roles', [])
            ))
        )

        # 8. Durations
        tasks.append(
            ('durations', match_single_filter(
                'durations',
                clauses.get('duration'),
                filter_options.get('durations', [])
            ))
        )

        # 9-12: These filters typically don't need AI matching (certificate_numbers, ranges)
        # We'll skip them but keep the structure

        # Execute all tasks in parallel
        results = await asyncio.gather(*[task[1] for task in tasks], return_exceptions=True)

        # Build matched_filters and confidence_scores
        matched_filters = {}
        confidence_scores = {}

        for (filter_name, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                logger.error(f"❌ Task failed for {filter_name}: {result}")
                # Don't add to matched_filters if failed
                continue
            elif result is not None:
                matches, confidence = result
                # Only add to matched_filters if we have actual matches
                if matches and len(matches) > 0:
                    matched_filters[filter_name] = matches
                    confidence_scores[filter_name] = confidence
                    logger.info(f"✅ Added {filter_name}: {matches} (confidence: {confidence})")
            else:
                logger.warning(f"⚠️ {filter_name} returned None (likely no extracted value)")

        # Handle range filters (no AI needed, direct from clauses)
        if clauses.get('consultancy_fee_min') or clauses.get('consultancy_fee_max'):
            matched_filters['consultancy_fee_range'] = {
                'min': clauses.get('consultancy_fee_min'),
                'max': clauses.get('consultancy_fee_max')
            }
            confidence_scores['consultancy_fee_range'] = 1.0

        if clauses.get('project_value_min') or clauses.get('project_value_max'):
            matched_filters['project_value_range'] = {
                'min': clauses.get('project_value_min'),
                'max': clauses.get('project_value_max')
            }
            confidence_scores['project_value_range'] = 1.0

        if clauses.get('completion_date_start') or clauses.get('completion_date_end'):
            matched_filters['completion_date_range'] = {
                'start': clauses.get('completion_date_start'),
                'end': clauses.get('completion_date_end')
            }
            confidence_scores['completion_date_range'] = 1.0

        logger.info(f"✅ Filter matching complete: {len(matched_filters)} filters matched")
        logger.info(f"Matched filters summary: {list(matched_filters.keys())}")
        for key, val in matched_filters.items():
            if isinstance(val, list):
                logger.info(f"  - {key}: {val[:3]}{'...' if len(val) > 3 else ''}")
            else:
                logger.info(f"  - {key}: {val}")

        return {
            "success": True,
            "matched_filters": matched_filters,
            "confidence_scores": confidence_scores
        }

    except json.JSONDecodeError as e:
        logger.error(f"❌ Failed to parse AI response: {e}")
        raise HTTPException(status_code=500, detail="AI returned invalid JSON")

    except OpenAIServiceError as e:
        logger.error(f"❌ OpenAI API error: {e}")
        raise HTTPException(status_code=500, detail=f"AI service error: {str(e)}")

    except Exception as e:
        logger.error(f"❌ Error in filter matching: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Matching error: {str(e)}")


@router.post("/api/certificates/search")
@require_company_details
async def search_certificates_post(
    request: Request,
    db: Session = Depends(get_db)
):
    """Search certificates with UNIVERSAL search across ALL fields - text, dates, numbers, JSONB arrays, etc."""
    current_user = get_current_user(request, db)
    if not current_user:
        logger.error("❌ Certificate search: User not authenticated")
        raise HTTPException(status_code=401, detail="Authentication required")

    logger.info(f"🔍 Certificate search: User {current_user.email} (ID: {current_user.id})")

    # Parse request body with pagination support
    try:
        body = await request.json()
        query = body.get("query", "").strip()
        filter_type = body.get("filter", "all")
        page = body.get("page", 1)
        per_page = body.get("per_page", 30)  # Default 30 certificates per page
        filters = body.get("filters", {})
        
        # Ensure valid pagination values
        page = max(1, int(page))
        per_page = max(1, min(100, int(per_page)))  # Cap at 100 for performance
        
        logger.info(f"🔍 Search query: '{query}', filter: {filter_type}, page: {page}, per_page: {per_page}")
        if filters:
            logger.info(f"🎯 Filters received: {json.dumps(filters, indent=2)}")
    except Exception as e:
        logger.error(f"❌ Failed to parse request body: {e}")
        query = ""
        filter_type = "all"
        page = 1
        per_page = 30
        filters = {}

    # Build base query
    filter_conditions = [
        CertificateDB.user_id == current_user.id,
        CertificateDB.processing_status == "completed"
    ]

    # Apply filters if provided
    filters_applied = False
    if filters:
        filters_applied = True

        # Client names filter (multi-select)
        if filters.get("clients"):
            filter_conditions.append(CertificateDB.client_name.in_(filters["clients"]))
            logger.info(f"📊 Applying client filter: {len(filters['clients'])} clients")

        # Locations filter (multi-select)
        if filters.get("locations"):
            filter_conditions.append(CertificateDB.location.in_(filters["locations"]))
            logger.info(f"📍 Applying location filter: {len(filters['locations'])} locations")

        # Consultancy fee range filter (using numeric column for efficient SQL filtering)
        if filters.get("consultancy_fee_range"):
            fee_range = filters["consultancy_fee_range"]
            min_fee = fee_range.get("min")
            max_fee = fee_range.get("max")

            if min_fee is not None:
                filter_conditions.append(CertificateDB.consultancy_fee_numeric >= min_fee)
                logger.info(f"💰 Applying fee range min: ₹{min_fee:,.2f}")

            if max_fee is not None:
                filter_conditions.append(CertificateDB.consultancy_fee_numeric <= max_fee)
                logger.info(f"💰 Applying fee range max: ₹{max_fee:,.2f}")

        # Project value range filter
        if filters.get("project_value_range"):
            pv_range = filters["project_value_range"]
            if pv_range.get("min") is not None:
                filter_conditions.append(CertificateDB.project_value >= pv_range["min"])
                logger.info(f"💼 Applying project value min: {pv_range['min']}")
            if pv_range.get("max") is not None:
                filter_conditions.append(CertificateDB.project_value <= pv_range["max"])
                logger.info(f"💼 Applying project value max: {pv_range['max']}")

        # Completion date range filter
        if filters.get("completion_date_range"):
            date_range = filters["completion_date_range"]
            if date_range.get("start"):
                try:
                    start_date = datetime.fromisoformat(date_range["start"].replace("Z", "+00:00"))
                    filter_conditions.append(CertificateDB.completion_date >= start_date)
                    logger.info(f"📅 Applying completion date start: {start_date}")
                except Exception as e:
                    logger.warning(f"⚠️ Invalid start date format: {date_range['start']}, error: {e}")
            if date_range.get("end"):
                try:
                    end_date = datetime.fromisoformat(date_range["end"].replace("Z", "+00:00"))
                    filter_conditions.append(CertificateDB.completion_date <= end_date)
                    logger.info(f"📅 Applying completion date end: {end_date}")
                except Exception as e:
                    logger.warning(f"⚠️ Invalid end date format: {date_range['end']}, error: {e}")

        # Services filter (JSONB array - PostgreSQL @> containment operator for exact match)
        if filters.get("services"):
            services_conditions = []
            for service in filters["services"]:
                # Use PostgreSQL JSONB @> containment operator for efficient exact array element matching
                # jsonb_build_array() creates a proper JSONB array from the service string
                services_conditions.append(
                    CertificateDB.services_rendered.op('@>')(func.jsonb_build_array(service))
                )
            if services_conditions:
                filter_conditions.append(or_(*services_conditions))
                logger.info(f"🔧 Applying services filter: {len(filters['services'])} services")

        # Sectors filter (JSONB array - PostgreSQL @> containment operator for exact match)
        if filters.get("sectors"):
            sectors_conditions = []
            for sector in filters["sectors"]:
                # Use PostgreSQL JSONB @> containment operator for efficient exact array element matching
                # jsonb_build_array() creates a proper JSONB array from the sector string
                sectors_conditions.append(
                    CertificateDB.sectors.op('@>')(func.jsonb_build_array(sector))
                )
            if sectors_conditions:
                filter_conditions.append(or_(*sectors_conditions))
                logger.info(f"🏭 Applying sectors filter: {len(filters['sectors'])} sectors")

        # Sub-sectors filter (JSONB array - PostgreSQL @> containment operator for exact match)
        if filters.get("sub_sectors"):
            subsectors_conditions = []
            for subsector in filters["sub_sectors"]:
                # Use PostgreSQL JSONB @> containment operator for efficient exact array element matching
                # jsonb_build_array() creates a proper JSONB array from the subsector string
                subsectors_conditions.append(
                    CertificateDB.sub_sectors.op('@>')(func.jsonb_build_array(subsector))
                )
            if subsectors_conditions:
                filter_conditions.append(or_(*subsectors_conditions))
                logger.info(f"🏗️ Applying sub-sectors filter: {len(filters['sub_sectors'])} sub-sectors")

        # Funding agency filter (multi-select)
        if filters.get("funding_agencies"):
            filter_conditions.append(CertificateDB.funding_agency.in_(filters["funding_agencies"]))
            logger.info(f"🏦 Applying funding agency filter: {len(filters['funding_agencies'])} agencies")

    # Log filter summary
    filter_count = len(filter_conditions) - 2  # Subtract user_id and processing_status base conditions
    logger.info(f"📊 Total filter conditions applied: {filter_count} (plus 2 base conditions)")

    base_query = db.query(CertificateDB).filter(and_(*filter_conditions))

    # If no search query, return all certificates for the user with pagination
    if not query:
        import time
        start_time = time.time()
        
        # Get total count BEFORE applying pagination
        total_count = base_query.count()
        
        # Calculate pagination metadata
        total_pages = (total_count + per_page - 1) // per_page if total_count > 0 else 1
        offset = (page - 1) * per_page
        
        # Apply pagination with offset and limit
        all_certificates = base_query.order_by(desc(CertificateDB.created_at)).offset(offset).limit(per_page).all()
        query_time = time.time() - start_time

        # Note: Consultancy fee filter is now applied in SQL using consultancy_fee_numeric column
        # No need for Python-side filtering anymore

        logger.info(f"✅ Query executed in {query_time:.3f}s, returned {len(all_certificates)} of {total_count} certificates (page {page}/{total_pages}, filters_applied: {filters_applied})")

        # Format results
        results = []
        for cert in all_certificates:
            results.append({
                "id": cert.id,
                "project_name": cert.project_name,
                "client_name": cert.client_name,
                "completion_date": cert.completion_date.isoformat() if cert.completion_date else None,
                "project_value": cert.project_value,
                "services_rendered": cert.services_rendered,
                "location": cert.location,
                "original_filename": cert.original_filename,
                "created_at": cert.created_at.isoformat() if cert.created_at else None
            })

        return {
            "certificates": results,
            "total": total_count,
            "page": page,
            "per_page": per_page,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1,
            "filters_applied": filters_applied
        }

    # UNIVERSAL SEARCH - Search EVERY field in the certificate database
    search_term = f"%{query}%"
    logger.info(f"🔍 Performing universal search across all certificate fields")

    filtered_query = base_query.filter(
        or_(
            # Core text fields
            CertificateDB.project_name.ilike(search_term),
            CertificateDB.client_name.ilike(search_term),
            CertificateDB.location.ilike(search_term),
            CertificateDB.original_filename.ilike(search_term),

            # Large text fields - Full certificate content
            CertificateDB.extracted_text.ilike(search_term),
            CertificateDB.verbatim_certificate.ilike(search_term),
            CertificateDB.scope_of_work.ilike(search_term),
            CertificateDB.issuing_authority_details.ilike(search_term),
            CertificateDB.performance_remarks.ilike(search_term),
            CertificateDB.signing_authority_details.ilike(search_term),

            # Reference and metadata fields
            CertificateDB.certificate_number.ilike(search_term),
            CertificateDB.role_lead_jv.ilike(search_term),
            CertificateDB.funding_agency.ilike(search_term),
            CertificateDB.duration.ilike(search_term),

            # Financial fields
            CertificateDB.consultancy_fee_inr.ilike(search_term),
            CertificateDB.project_value_inr.ilike(search_term),
            cast(CertificateDB.project_value, String).ilike(search_term),

            # JSONB array fields (cast to text for searching)
            cast(CertificateDB.services_rendered, String).ilike(search_term),
            cast(CertificateDB.sectors, String).ilike(search_term),
            cast(CertificateDB.sub_sectors, String).ilike(search_term),
            cast(CertificateDB.jv_partners, String).ilike(search_term),
            cast(CertificateDB.metrics, String).ilike(search_term),

            # Date fields (cast to text for date string matching)
            cast(CertificateDB.completion_date, String).ilike(search_term),
            cast(CertificateDB.start_date, String).ilike(search_term),
            cast(CertificateDB.end_date, String).ilike(search_term)
        )
    )

    # Get total count BEFORE applying pagination
    total_count = filtered_query.count()
    
    # Calculate pagination metadata
    total_pages = (total_count + per_page - 1) // per_page if total_count > 0 else 1
    offset = (page - 1) * per_page
    
    # Execute the search with pagination
    certificates = filtered_query.order_by(desc(CertificateDB.created_at)).offset(offset).limit(per_page).all()

    # Note: Consultancy fee filter is now applied in SQL using consultancy_fee_numeric column
    # No need for Python-side filtering anymore

    logger.info(f"✅ Universal search returned {len(certificates)} of {total_count} certificates for query: '{query}' (page {page}/{total_pages}, filters_applied: {filters_applied})")

    # Format results with attachment information
    from database import TenderCertificateAttachmentDB

    results = []
    for cert in certificates:
        # Get attachment info for this certificate
        attachments = db.query(TenderCertificateAttachmentDB).filter(
            TenderCertificateAttachmentDB.certificate_id == cert.id,
            TenderCertificateAttachmentDB.attached_by_user_id == current_user.id
        ).all()

        attached_tenders = []
        for att in attachments:
            if att.tender:
                attached_tenders.append({
                    "tender_id": att.tender_id,
                    "tender_title": att.tender.title
                })

        results.append({
            "id": cert.id,
            "project_name": cert.project_name,
            "client_name": cert.client_name,
            "completion_date": cert.completion_date.isoformat() if cert.completion_date else None,
            "project_value": cert.project_value,
            "services_rendered": cert.services_rendered,
            "location": cert.location,
            "original_filename": cert.original_filename,
            "created_at": cert.created_at.isoformat() if cert.created_at else None,
            # Attachment information
            "attached_tenders_count": len(attachments),
            "is_attached": len(attachments) > 0,
            "attached_tenders": attached_tenders
        })

    return {
        "certificates": results,
        "total": total_count,
        "page": page,
        "per_page": per_page,
        "total_pages": total_pages,
        "has_next": page < total_pages,
        "has_prev": page > 1,
        "filters_applied": filters_applied
    }


@router.get("/api/certificates/batch/{batch_id}/status")
@require_company_details
async def get_batch_status(
    request: Request,
    batch_id: str,
    db: Session = Depends(get_db)
):
    """Get the status of a certificate batch upload."""
    current_user = get_current_user(request, db)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Get batch record
    batch = db.query(BulkUploadBatchDB).filter(
        and_(
            BulkUploadBatchDB.id == batch_id,
            BulkUploadBatchDB.user_id == current_user.id
        )
    ).first()

    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    # Get certificates in this batch
    certificates = db.query(CertificateDB).filter(
        CertificateDB.batch_id == batch_id
    ).all()

    # Calculate progress percentage
    progress_percentage = 0
    if batch.total_files > 0:
        progress_percentage = int((batch.processed_count / batch.total_files) * 100)

    return {
        "batch": batch.to_dict(),
        "progress_percentage": progress_percentage,
        "certificates": [
            {
                "id": cert.id,
                "filename": cert.original_filename,
                "status": cert.processing_status,
                "project_name": cert.project_name,
                "error": cert.processing_error
            }
            for cert in certificates
        ],
        "queue_status": get_queue_status()
    }


@router.get("/api/certificates/batches")
@require_company_details
async def get_user_batches(
    request: Request,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Get all batch uploads for the current user."""
    current_user = get_current_user(request, db)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Get recent batches
    batches = db.query(BulkUploadBatchDB).filter(
        BulkUploadBatchDB.user_id == current_user.id
    ).order_by(BulkUploadBatchDB.created_at.desc()).limit(limit).all()

    return {
        "batches": [batch.to_dict() for batch in batches],
        "queue_status": get_queue_status()
    }


@router.delete("/api/certificates/batch/{batch_id}")
@require_company_details
async def delete_batch(
    request: Request,
    batch_id: str,
    db: Session = Depends(get_db)
):
    """Delete a batch and all its certificates."""
    current_user = get_current_user(request, db)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Get batch record
    batch = db.query(BulkUploadBatchDB).filter(
        and_(
            BulkUploadBatchDB.id == batch_id,
            BulkUploadBatchDB.user_id == current_user.id
        )
    ).first()

    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    # Get all certificates in batch
    certificates = db.query(CertificateDB).filter(
        CertificateDB.batch_id == batch_id
    ).all()

    # Delete each certificate with cleanup
    for cert in certificates:
        certificate_processor.delete_certificate_with_cleanup(cert.id, current_user.id)

    # Delete batch record
    db.delete(batch)
    db.commit()

    return {
        "message": f"Deleted batch {batch_id} and {len(certificates)} certificates"
    }


# ============================================================================
# FAILED CERTIFICATE TASK MANAGEMENT
# ============================================================================

@router.get("/api/certificates/failed-tasks")
@require_company_details
async def get_failed_certificate_tasks(
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Get all permanently failed certificate processing tasks.
    These are tasks that have exhausted all retry attempts.
    """
    current_user = get_current_user(request, db)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    failed_tasks = get_failed_tasks()

    # Filter to only show tasks belonging to this user
    user_tasks = [
        task for task in failed_tasks
        if task.get('user_id') == current_user.id
    ]

    return {
        "failed_tasks": user_tasks,
        "total_count": len(user_tasks),
        "queue_status": get_queue_status()
    }


@router.post("/api/certificates/failed-tasks/{task_id}/retry")
@require_company_details
async def retry_failed_certificate_task(
    request: Request,
    task_id: str,
    db: Session = Depends(get_db)
):
    """
    Retry a failed certificate processing task.
    The task will be re-queued for processing with reset retry count.
    """
    current_user = get_current_user(request, db)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Verify the task belongs to this user
    failed_tasks = get_failed_tasks()
    task = next(
        (t for t in failed_tasks if t.get('task_id') == task_id),
        None
    )

    if not task:
        raise HTTPException(status_code=404, detail="Failed task not found")

    if task.get('user_id') != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to retry this task")

    success = retry_failed_task(task_id)

    if success:
        return {
            "message": f"Task {task_id} re-queued for processing",
            "task_id": task_id,
            "filename": task.get('filename')
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to re-queue task")


@router.post("/api/certificates/failed-tasks/retry-all")
@require_company_details
async def retry_all_failed_certificate_tasks(
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Retry all failed certificate processing tasks for the current user.
    """
    current_user = get_current_user(request, db)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    failed_tasks = get_failed_tasks()

    # Filter to only user's tasks
    user_tasks = [
        task for task in failed_tasks
        if task.get('user_id') == current_user.id
    ]

    retried_count = 0
    for task in user_tasks:
        task_id = task.get('task_id')
        if task_id and retry_failed_task(task_id):
            retried_count += 1

    return {
        "message": f"Re-queued {retried_count} failed tasks for processing",
        "retried_count": retried_count,
        "total_failed": len(user_tasks)
    }


@router.get("/api/certificates/queue-status")
@require_company_details
async def get_certificate_queue_status(
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Get detailed status of the certificate processing queue.
    Includes queue depth, active workers, retry queue, and failed tasks.
    """
    current_user = get_current_user(request, db)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    queue_status = get_queue_status()
    failed_tasks = get_failed_tasks()

    # Count user's failed tasks
    user_failed_count = len([
        task for task in failed_tasks
        if task.get('user_id') == current_user.id
    ])

    return {
        "queue_status": queue_status,
        "user_failed_count": user_failed_count,
        "redis_connected": queue_status.get('redis_connected', False),
        "processing_active": queue_status.get('is_running', False)
    }


# ============================================================================
# FAVORITES LIST FOR CERTIFICATE ATTACHMENT
# ============================================================================

@router.get("/api/favorites/list-for-attachment")
async def get_favorites_for_attachment(
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Get simplified list of favorited tenders for certificate attachment dropdown.
    Only includes active (non-awarded) tenders.

    Returns:
        JSON list of favorited tenders with minimal fields
    """
    from core.dependencies import get_id_for_tender_management
    from database import FavoriteDB
    from sqlalchemy.orm import joinedload
    from sqlalchemy import desc

    # Get authenticated entity
    entity_id, entity, entity_type = get_id_for_tender_management(request, db)
    if not entity_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Query favorites with tender details
    favorites = db.query(FavoriteDB).options(
        joinedload(FavoriteDB.tender)
    ).filter(
        FavoriteDB.user_id == entity_id
    ).order_by(desc(FavoriteDB.created_at)).all()

    # Filter and format
    now = datetime.utcnow()
    result = []

    for fav in favorites:
        if not fav.tender:
            continue

        # Skip awarded tenders
        if fav.tender.awarded:
            continue

        # Skip expired tenders
        if fav.tender.deadline and fav.tender.deadline < now:
            continue

        result.append({
            "tender_id": fav.tender.id,
            "title": fav.tender.title,
            "authority": fav.tender.authority,
            "deadline": fav.tender.deadline.isoformat() if fav.tender.deadline else None,
            "reference_number": fav.tender.tender_reference_number,
            "estimated_value": fav.tender.estimated_value,
            "favorited_at": fav.created_at.isoformat()
        })

    return {
        "favorites": result,
        "count": len(result)
    }


# ============================================================================
# CERTIFICATE-TENDER ATTACHMENT ENDPOINTS
# ============================================================================

@router.post("/api/certificates/{certificate_id}/attach-to-tender")
@require_company_details
async def attach_certificate_to_tender(
    request: Request,
    certificate_id: str,
    tender_id: str = Form(...),
    notes: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Attach a certificate to a favorited tender.

    Args:
        certificate_id: UUID of the certificate
        tender_id: ID of the tender (must be in user's favorites)
        notes: Optional notes about why this certificate is relevant

    Returns:
        JSON with attachment details
    """
    from core.dependencies import get_id_for_tender_management
    from database import TenderCertificateAttachmentDB, FavoriteDB, CertificateDB, TenderDB

    # Get authenticated entity
    entity_id, entity, entity_type = get_id_for_tender_management(request, db)
    if not entity_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Verify certificate exists and belongs to user
    certificate = db.query(CertificateDB).filter(
        CertificateDB.id == certificate_id,
        CertificateDB.user_id == entity_id
    ).first()

    if not certificate:
        raise HTTPException(status_code=404, detail="Certificate not found or access denied")

    # Verify certificate is processed
    if certificate.processing_status != "completed":
        raise HTTPException(status_code=400, detail="Certificate is not fully processed yet")

    # Verify tender exists
    tender = db.query(TenderDB).filter(TenderDB.id == tender_id).first()
    if not tender:
        raise HTTPException(status_code=404, detail="Tender not found")

    # Verify tender is in user's favorites
    favorite = db.query(FavoriteDB).filter(
        FavoriteDB.user_id == entity_id,
        FavoriteDB.tender_id == tender_id
    ).first()

    if not favorite:
        raise HTTPException(status_code=400, detail="You can only attach certificates to your favorited tenders")

    # Check if already attached
    existing_attachment = db.query(TenderCertificateAttachmentDB).filter(
        TenderCertificateAttachmentDB.certificate_id == certificate_id,
        TenderCertificateAttachmentDB.tender_id == tender_id
    ).first()

    if existing_attachment:
        raise HTTPException(status_code=400, detail="Certificate is already attached to this tender")

    # Create attachment
    attachment = TenderCertificateAttachmentDB(
        certificate_id=certificate_id,
        tender_id=tender_id,
        attached_by_user_id=entity_id,
        attached_by_type=entity_type,
        attached_by_name=entity.name,
        notes=notes
    )

    db.add(attachment)
    db.commit()
    db.refresh(attachment)

    logger.info(f"[CERT ATTACH] {entity_type} {entity.name} attached cert {certificate_id[:8]} to tender {tender_id[:8]}")

    return {
        "success": True,
        "message": "Certificate attached to tender successfully",
        "attachment_id": attachment.id,
        "certificate_id": certificate_id,
        "tender_id": tender_id,
        "tender_title": tender.title,
        "attached_at": attachment.attached_at.isoformat()
    }


@router.delete("/api/certificates/{certificate_id}/detach-from-tender/{tender_id}")
@require_company_details
async def detach_certificate_from_tender(
    request: Request,
    certificate_id: str,
    tender_id: str,
    db: Session = Depends(get_db)
):
    """
    Detach a certificate from a tender.

    Args:
        certificate_id: UUID of the certificate
        tender_id: ID of the tender

    Returns:
        JSON confirmation
    """
    from core.dependencies import get_id_for_tender_management
    from database import TenderCertificateAttachmentDB

    # Get authenticated entity
    entity_id, entity, entity_type = get_id_for_tender_management(request, db)
    if not entity_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Find attachment
    attachment = db.query(TenderCertificateAttachmentDB).filter(
        TenderCertificateAttachmentDB.certificate_id == certificate_id,
        TenderCertificateAttachmentDB.tender_id == tender_id,
        TenderCertificateAttachmentDB.attached_by_user_id == entity_id
    ).first()

    if not attachment:
        raise HTTPException(status_code=404, detail="Attachment not found or access denied")

    db.delete(attachment)
    db.commit()

    logger.info(f"[CERT DETACH] {entity_type} {entity.name} detached cert {certificate_id[:8]} from tender {tender_id[:8]}")

    return {
        "success": True,
        "message": "Certificate detached from tender successfully"
    }


@router.get("/api/certificates/{certificate_id}/attached-tenders")
@require_company_details
async def get_attached_tenders_for_certificate(
    request: Request,
    certificate_id: str,
    db: Session = Depends(get_db)
):
    """
    Get all tenders that a certificate is attached to.

    Args:
        certificate_id: UUID of the certificate

    Returns:
        JSON list of attached tenders
    """
    from core.dependencies import get_id_for_tender_management
    from database import TenderCertificateAttachmentDB
    from sqlalchemy.orm import joinedload

    # Get authenticated entity
    entity_id, entity, entity_type = get_id_for_tender_management(request, db)
    if not entity_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Query attachments with eager loading
    attachments = db.query(TenderCertificateAttachmentDB).options(
        joinedload(TenderCertificateAttachmentDB.tender)
    ).filter(
        TenderCertificateAttachmentDB.certificate_id == certificate_id,
        TenderCertificateAttachmentDB.attached_by_user_id == entity_id
    ).all()

    result = []
    for att in attachments:
        if att.tender:
            result.append({
                "attachment_id": att.id,
                "tender_id": att.tender.id,
                "tender_title": att.tender.title,
                "tender_authority": att.tender.authority,
                "tender_deadline": att.tender.deadline.isoformat() if att.tender.deadline else None,
                "attached_at": att.attached_at.isoformat(),
                "notes": att.notes
            })

    return {"attachments": result, "count": len(result)}


@router.get("/api/tenders/{tender_id}/attached-certificates")
async def get_attached_certificates_for_tender(
    request: Request,
    tender_id: str,
    db: Session = Depends(get_db)
):
    """
    Get all certificates attached to a tender (works for any lifecycle stage).

    Args:
        tender_id: ID of the tender (or source_tender_id for projects)

    Returns:
        JSON list of attached certificates
    """
    from core.dependencies import get_id_for_tender_management
    from database import TenderCertificateAttachmentDB
    from sqlalchemy.orm import joinedload

    # Get authenticated entity
    entity_id, entity, entity_type = get_id_for_tender_management(request, db)
    if not entity_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Query attachments with eager loading
    attachments = db.query(TenderCertificateAttachmentDB).options(
        joinedload(TenderCertificateAttachmentDB.certificate)
    ).filter(
        TenderCertificateAttachmentDB.tender_id == tender_id
    ).all()

    result = []
    for att in attachments:
        if att.certificate:
            cert = att.certificate
            result.append({
                "attachment_id": att.id,
                "certificate_id": cert.id,
                "project_name": cert.project_name,
                "client_name": cert.client_name,
                "completion_date": cert.completion_date.isoformat() if cert.completion_date else None,
                "project_value": cert.project_value,
                "project_value_inr": cert.project_value_inr,
                "consultancy_fee": cert.consultancy_fee_numeric,
                "consultancy_fee_inr": cert.consultancy_fee_inr,
                "services_rendered": cert.services_rendered or [],
                "attached_at": att.attached_at.isoformat(),
                "attached_by": att.attached_by_name,
                "notes": att.notes
            })

    return {"certificates": result, "count": len(result)}
