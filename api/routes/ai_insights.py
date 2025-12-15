"""
AI Insights API Routes
Endpoints for extracting and retrieving AI-powered tender insights
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session
from datetime import datetime
import json
import io

from core.dependencies import get_current_user, require_company_details
from core.ai_insights_extractor import AIInsightsExtractor
from database import get_db, TenderDB, TenderAIInsightsDB, TenderDocumentDB
import os

router = APIRouter()


def _require_user(request: Request, db: Session):
    current_user = get_current_user(request, db)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return current_user


@router.post("/api/tenders/{tender_id}/extract-insights")
@require_company_details
async def extract_ai_insights(
    tender_id: str,
    request: Request,
    force: bool = Query(False, description="Force re-extraction"),
    document_id: int = Query(None, description="Specific document ID to analyze"),
    db: Session = Depends(get_db),
):
    """
    Trigger AI insights extraction for a tender document
    Extracts PQ and Eligibility criteria using GPT-4o
    """
    user = _require_user(request, db)

    # Verify tender exists
    tender = db.query(TenderDB).filter(TenderDB.id == tender_id).first()
    if not tender:
        raise HTTPException(status_code=404, detail="Tender not found")

    # Get OpenAI API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OpenAI API key not configured"
        )

    # Initialize extractor
    extractor = AIInsightsExtractor(api_key)

    try:
        result = extractor.extract_insights(tender_id, db, force=force, document_id=document_id)
        return JSONResponse(result)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Extraction failed: {str(e)}"
        )


@router.get("/api/tenders/{tender_id}/insights-status")
@require_company_details
async def get_insights_status(
    tender_id: str,
    request: Request,
    db: Session = Depends(get_db),
):
    """
    Get current AI insights extraction status
    """
    _require_user(request, db)

    insights = db.query(TenderAIInsightsDB).filter(
        TenderAIInsightsDB.tender_id == tender_id
    ).first()

    if not insights:
        return JSONResponse({
            "tender_id": tender_id,
            "status": "not_started",
            "total_criteria": 0
        })

    return JSONResponse({
        "tender_id": tender_id,
        "insights_id": insights.id,
        "status": insights.extraction_status,
        "document_filename": insights.document_filename,
        "document_page_count": insights.document_page_count,
        "total_pq_criteria": insights.total_pq_criteria,
        "total_eligibility_criteria": insights.total_eligibility_criteria,
        "total_criteria": insights.total_pq_criteria + insights.total_eligibility_criteria,
        "mandatory_count": insights.mandatory_count,
        "optional_count": insights.optional_count,
        "error": insights.error_message,
        "started_at": insights.started_at.isoformat() if insights.started_at else None,
        "completed_at": insights.completed_at.isoformat() if insights.completed_at else None,
        "raw_gpt_response": insights.raw_gpt_response,
    })


@router.get("/api/tenders/{tender_id}/insights-criteria")
@require_company_details
async def get_insights_criteria(
    tender_id: str,
    request: Request,
    category: str = Query(None, description="Filter by category: pq or eligibility"),
    db: Session = Depends(get_db),
):
    """
    Get extracted criteria from AI insights
    """
    _require_user(request, db)

    insights = db.query(TenderAIInsightsDB).filter(
        TenderAIInsightsDB.tender_id == tender_id
    ).first()

    if not insights:
        return JSONResponse({
            "tender_id": tender_id,
            "has_insights": False,
            "criteria": []
        })

    # Combine criteria based on filter
    if category == "pq":
        criteria = insights.pq_criteria or []
    elif category == "eligibility":
        criteria = insights.eligibility_criteria or []
    else:
        # Return both with category labels
        pq_criteria = [
            {**c, "category": "pq"}
            for c in (insights.pq_criteria or [])
        ]
        eligibility_criteria = [
            {**c, "category": "eligibility"}
            for c in (insights.eligibility_criteria or [])
        ]
        criteria = pq_criteria + eligibility_criteria

    return JSONResponse({
        "tender_id": tender_id,
        "insights_id": insights.id,
        "has_insights": True,
        "status": insights.extraction_status,
        "total_criteria": len(criteria),
        "criteria": criteria,
        "summary": insights.summary or {},
        "raw_gpt_response": insights.raw_gpt_response
    })


@router.get("/api/tenders/{tender_id}/insights-download")
@require_company_details
async def download_insights_report(
    tender_id: str,
    request: Request,
    db: Session = Depends(get_db),
):
    """
    Download AI insights as JSON report
    """
    _require_user(request, db)

    insights = db.query(TenderAIInsightsDB).filter(
        TenderAIInsightsDB.tender_id == tender_id
    ).first()

    if not insights:
        raise HTTPException(status_code=404, detail="No insights found")

    # Build report
    report = {
        "tender_id": tender_id,
        "extraction_date": insights.completed_at.isoformat() if insights.completed_at else None,
        "document_filename": insights.document_filename,
        "document_page_count": insights.document_page_count,
        "summary": insights.summary or {},
        "pq_criteria": insights.pq_criteria or [],
        "eligibility_criteria": insights.eligibility_criteria or [],
        "sections": insights.sections or []
    }

    # Convert to JSON
    json_str = json.dumps(report, indent=2)
    json_bytes = json_str.encode('utf-8')

    # Create streaming response
    return StreamingResponse(
        io.BytesIO(json_bytes),
        media_type="application/json",
        headers={
            "Content-Disposition": f"attachment; filename=tender_{tender_id}_insights.json"
        }
    )


@router.get("/api/tenders/{tender_id}/pdf-documents")
@require_company_details
async def get_tender_pdf_documents(
    tender_id: str,
    request: Request,
    db: Session = Depends(get_db),
):
    """
    Get list of all PDF documents for a tender
    Used for document selection in AI insights extraction
    """
    _require_user(request, db)

    # Get all PDF documents for this tender
    documents = db.query(TenderDocumentDB).filter(
        TenderDocumentDB.tender_id == tender_id,
        TenderDocumentDB.document_type == "pdf"
    ).order_by(TenderDocumentDB.file_size.desc()).all()

    if not documents:
        return JSONResponse({
            "tender_id": tender_id,
            "documents": []
        })

    # Format document list
    doc_list = []
    for doc in documents:
        doc_list.append({
            "id": doc.id,
            "filename": doc.filename,
            "file_size": doc.file_size,
            "file_size_mb": round(doc.file_size / (1024 * 1024), 2),
            "mime_type": doc.mime_type,
            "created_at": doc.created_at.isoformat() if doc.created_at else None
        })

    return JSONResponse({
        "tender_id": tender_id,
        "total_documents": len(doc_list),
        "documents": doc_list
    })


@router.get("/api/tenders/{tender_id}/insights-debug")
@require_company_details
async def debug_insights_extraction(
    tender_id: str,
    request: Request,
    db: Session = Depends(get_db),
):
    """
    Debug endpoint to see extraction details and quality metrics
    Returns: text_length, was_truncated, warnings, raw_gpt_response, errors

    This endpoint is useful for troubleshooting blank or incomplete extractions.
    """
    _require_user(request, db)

    insights = db.query(TenderAIInsightsDB).filter(
        TenderAIInsightsDB.tender_id == tender_id
    ).first()

    if not insights:
        raise HTTPException(status_code=404, detail="No insights found for this tender")

    return JSONResponse({
        "tender_id": tender_id,
        "insights_id": insights.id,
        "extraction_status": insights.extraction_status,
        "document_info": {
            "filename": insights.document_filename,
            "page_count": insights.document_page_count,
            "text_length": insights.text_length,
            "was_truncated": insights.was_truncated,
        },
        "extraction_quality": {
            "warnings": insights.extraction_warnings or [],
            "error_message": insights.error_message,
        },
        "criteria_counts": {
            "pq": insights.total_pq_criteria,
            "eligibility": insights.total_eligibility_criteria,
            "mandatory": insights.mandatory_count,
            "optional": insights.optional_count,
            "total": insights.total_pq_criteria + insights.total_eligibility_criteria,
        },
        "raw_gpt_response_preview": insights.raw_gpt_response[:1000] if insights.raw_gpt_response else None,
        "raw_gpt_response_length": len(insights.raw_gpt_response) if insights.raw_gpt_response else 0,
        "timestamps": {
            "started_at": insights.started_at.isoformat() if insights.started_at else None,
            "completed_at": insights.completed_at.isoformat() if insights.completed_at else None,
            "created_at": insights.created_at.isoformat() if insights.created_at else None,
        }
    })
