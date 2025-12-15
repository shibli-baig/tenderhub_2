"""Placeholder routes for the certificate matching feature set."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from certificate_tender_matcher import certificate_tender_matcher
from core.dependencies import get_current_user, require_company_details
from database import (
    CertificateDB,
    TenderAnalysisStatusDB,
    TenderDB,
    TenderEligibilityCriteriaDB,
    TenderMatchDB,
    TenderMatchResultDB,
    get_db,
)
from tender_eligibility_queue import enqueue_tender_analysis, get_queue_status


logger = logging.getLogger(__name__)
router = APIRouter()


def _require_user(request: Request, db: Session):
    current_user = get_current_user(request, db)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return current_user


@router.post("/api/tenders/{tender_id}/analyze-eligibility")
@require_company_details
async def analyze_tender_eligibility(
    tender_id: str,
    request: Request,
    force: bool = Query(False, description="Set true to re-run analysis"),
    db: Session = Depends(get_db),
):
    """Queue tender eligibility extraction for the specified tender."""
    user = _require_user(request, db)

    tender = db.query(TenderDB).filter(TenderDB.id == tender_id).first()
    if not tender:
        raise HTTPException(status_code=404, detail="Tender not found")

    status = db.query(TenderAnalysisStatusDB).filter(
        TenderAnalysisStatusDB.tender_id == tender_id
    ).first()

    if status and status.status in {"processing", "queued"} and not force:
        return JSONResponse({
            "status": status.status,
            "analysis_status_id": status.id,
            "message": "Analysis already in progress",
            "progress": {
                "processed_pages": status.processed_pages,
                "total_pages": status.total_pages,
            },
            "queue": get_queue_status(),
        })

    if status and status.status == 'completed' and not force:
        return JSONResponse({
            "status": status.status,
            "analysis_status_id": status.id,
            "message": "Analysis already completed",
            "total_criteria": status.total_criteria_found,
        })

    if status and force:
        db.query(TenderEligibilityCriteriaDB).filter(
            TenderEligibilityCriteriaDB.tender_id == tender_id
        ).delete(synchronize_session=False)

    if status:
        analysis_status_id = status.id
        status.status = 'queued'
        status.total_pages = 0
        status.processed_pages = 0
        status.total_criteria_found = 0
        status.error_message = None
        status.started_at = None
        status.completed_at = None
        status.created_at = datetime.utcnow()
    else:
        analysis_status_id = str(uuid.uuid4())
        status = TenderAnalysisStatusDB(
            id=analysis_status_id,
            tender_id=tender_id,
            status='queued',
            created_at=datetime.utcnow(),
        )
        db.add(status)

    db.commit()

    task_id = enqueue_tender_analysis(
        tender_id=tender_id,
        user_id=user.id,
        analysis_status_id=analysis_status_id,
    )

    return JSONResponse({
        "status": "queued",
        "analysis_status_id": analysis_status_id,
        "task_id": task_id,
        "queue": get_queue_status(),
    })


@router.get("/api/tenders/{tender_id}/eligibility-status")
@require_company_details
async def get_eligibility_analysis_status(
    tender_id: str,
    request: Request,
    db: Session = Depends(get_db),
):
    """Return current tender analysis status."""
    _require_user(request, db)

    status = db.query(TenderAnalysisStatusDB).filter(
        TenderAnalysisStatusDB.tender_id == tender_id
    ).first()

    if not status:
        return JSONResponse({
            "tender_id": tender_id,
            "status": "not_started",
            "progress": {
                "processed_pages": 0,
                "total_pages": 0,
                "percentage": 0,
            }
        })

    percentage = 0
    if status.total_pages:
        percentage = round((status.processed_pages / status.total_pages) * 100, 1)

    return JSONResponse({
        "tender_id": tender_id,
        "analysis_status_id": status.id,
        "status": status.status,
        "total_criteria_found": status.total_criteria_found,
        "error": status.error_message,
        "progress": {
            "processed_pages": status.processed_pages,
            "total_pages": status.total_pages,
            "percentage": percentage,
        },
        "started_at": status.started_at.isoformat() if status.started_at else None,
        "completed_at": status.completed_at.isoformat() if status.completed_at else None,
    })


@router.get("/api/tenders/{tender_id}/eligibility-criteria")
@require_company_details
async def get_eligibility_criteria(
    tender_id: str,
    request: Request,
    category: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Return extracted eligibility criteria for the tender."""
    _require_user(request, db)

    query = db.query(TenderEligibilityCriteriaDB).filter(
        TenderEligibilityCriteriaDB.tender_id == tender_id
    ).order_by(TenderEligibilityCriteriaDB.page_number.asc())

    if category:
        query = query.filter(TenderEligibilityCriteriaDB.category == category)

    criteria = query.all()
    total = len(criteria)

    category_counts = (
        db.query(TenderEligibilityCriteriaDB.category, func.count(TenderEligibilityCriteriaDB.id))
        .filter(TenderEligibilityCriteriaDB.tender_id == tender_id)
        .group_by(TenderEligibilityCriteriaDB.category)
        .all()
    )
    categories = {cat: count for cat, count in category_counts}

    serialized = [
        {
            "id": c.id,
            "category": c.category,
            "type": c.criteria_type,
            "page_number": c.page_number,
            "text": c.criteria_text,
            "requirements": c.extracted_requirements,
            "keywords": c.keywords,
            "confidence": c.confidence_score,
        }
        for c in criteria
    ]

    return JSONResponse({
        "tender_id": tender_id,
        "category": category,
        "total_criteria": total,
        "categories": categories,
        "criteria": serialized,
    })


@router.post("/api/tenders/{tender_id}/match-certificates")
@require_company_details
async def match_certificates(
    tender_id: str,
    request: Request,
    db: Session = Depends(get_db),
):
    """Trigger certificate matching for the current user."""
    user = _require_user(request, db)
    mode = "keyword"
    try:
        payload = await request.json()
        if isinstance(payload, dict):
            mode = payload.get("mode", mode)
    except Exception:
        mode = "keyword"
    mode = request.query_params.get("mode", mode)

    try:
        result = certificate_tender_matcher.match_certificates_to_tender(tender_id, user.id, mode=mode)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - logged upstream
        logger.exception("Matching failed")
        raise HTTPException(status_code=500, detail="Certificate matching failed") from exc

    return JSONResponse(result)


@router.get("/api/certificates/{certificate_id}/tender-matches")
@require_company_details
async def get_certificate_tender_matches(
    certificate_id: str,
    request: Request,
    db: Session = Depends(get_db),
):
    """Fetch historical tender matches for a specific certificate."""
    user = _require_user(request, db)

    certificate = (
        db.query(CertificateDB)
        .filter(CertificateDB.id == certificate_id, CertificateDB.user_id == user.id)
        .first()
    )
    if not certificate:
        raise HTTPException(status_code=404, detail="Certificate not found")

    results = (
        db.query(TenderMatchResultDB)
        .join(TenderMatchDB, TenderMatchResultDB.match_id == TenderMatchDB.id)
        .filter(TenderMatchResultDB.certificate_id == certificate_id)
        .filter(TenderMatchDB.user_id == user.id)
        .order_by(TenderMatchResultDB.created_at.desc())
        .all()
    )

    response = {
        "certificate_id": certificate_id,
        "certificate_name": certificate.project_name,
        "total_matches": len(results),
        "tenders": [
            {
                "match_id": result.match_id,
                "tender_id": result.match.tender_id if result.match else None,
                "tender_title": result.match.tender_name if result.match else None,
                "score": result.score,
                "created_at": result.created_at.isoformat() if result.created_at else None,
            }
            for result in results
        ],
    }
    return JSONResponse(response)


@router.get("/api/tenders/{tender_id}/matching-summary")
@require_company_details
async def get_matching_summary(
    tender_id: str,
    request: Request,
    db: Session = Depends(get_db),
):
    """Return the latest matching summary for the tender."""
    user = _require_user(request, db)

    latest_match = (
        db.query(TenderMatchDB)
        .filter(TenderMatchDB.tender_id == tender_id, TenderMatchDB.user_id == user.id)
        .order_by(TenderMatchDB.created_at.desc())
        .first()
    )

    if not latest_match:
        return JSONResponse({
            "tender_id": tender_id,
            "user_id": user.id,
            "has_matches": False,
            "matched_certificates": 0,
            "total_certificates_considered": 0,
            "profile": None,
            "matching_method": "keyword",
        })

    summary = latest_match.summary or {}
    results = (
        db.query(TenderMatchResultDB)
        .filter(TenderMatchResultDB.match_id == latest_match.id)
        .order_by(TenderMatchResultDB.score.desc())
        .limit(25)
        .all()
    )

    matches = []
    for result in results:
        cert = result.certificate
        matches.append({
            "certificate_id": result.certificate_id,
            "project_name": cert.project_name if cert else None,
            "client_name": cert.client_name if cert else None,
            "location": cert.location if cert else None,
            "score": result.score,
            "matching_factors": result.matching_factors or [],
            "gaps": result.gaps or [],
        })

    return JSONResponse({
        "tender_id": tender_id,
        "user_id": user.id,
        "match_id": latest_match.id,
        "has_matches": bool(summary.get("matched_certificates")),
        "matched_certificates": summary.get("matched_certificates", 0),
        "total_certificates_considered": summary.get("total_certificates", 0),
        "best_score": summary.get("best_score", 0),
        "profile": summary.get("profile"),
        "matching_method": summary.get("matching_method", "keyword"),
        "created_at": latest_match.created_at.isoformat() if latest_match.created_at else None,
        "matches": matches,
    })
