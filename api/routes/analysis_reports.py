"""
API routes for tender analysis reports.
Handles employee analysis report creation, editing, file attachments, and user viewing.
"""

from fastapi import APIRouter, Request, Depends, HTTPException, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import and_
from datetime import datetime
from typing import Optional, Dict, Any
import json
import io

from database import (
    TenderDB, TenderAssignmentDB, TenderAnalysisReportDB,
    ReportAttachmentDB, EmployeeDB, TenderDocumentDB
)
from core.dependencies import get_db, get_current_employee, get_current_user

router = APIRouter()
templates = Jinja2Templates(directory="templates")

# File upload constraints
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_MIME_TYPES = [
    'application/pdf',
    'application/vnd.ms-excel',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'application/vnd.ms-powerpoint',
    'application/vnd.openxmlformats-officedocument.presentationml.presentation',
    'application/msword',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'image/png',
    'image/jpeg',
    'image/jpg'
]


@router.get("/employee/analysis/{tender_id}", response_class=HTMLResponse)
async def get_analysis_page(request: Request, tender_id: str, db: Session = Depends(get_db)):
    """
    Load employee analysis page for a tender.
    Shows existing report for editing or blank form for creation.
    """
    current_employee = get_current_employee(request, db)
    if not current_employee:
        return templates.TemplateResponse("employee_login.html", {"request": request})

    # Verify employee is assigned to this tender
    assignment = db.query(TenderAssignmentDB).filter(
        and_(
            TenderAssignmentDB.tender_id == tender_id,
            TenderAssignmentDB.employee_id == current_employee.id
        )
    ).first()

    if not assignment:
        raise HTTPException(status_code=403, detail="You are not assigned to this tender")

    # Get tender details
    tender = db.query(TenderDB).filter(TenderDB.id == tender_id).first()
    if not tender:
        raise HTTPException(status_code=404, detail="Tender not found")

    # Get existing report (if any)
    report = db.query(TenderAnalysisReportDB).filter(
        TenderAnalysisReportDB.tender_id == tender_id
    ).first()

    # Determine if current employee is the owner
    is_owner = False
    if report:
        is_owner = (report.employee_id == current_employee.id)
    else:
        # No report exists yet - current employee will be the owner if they create it
        is_owner = True

    # Get tender documents
    tender_documents = db.query(TenderDocumentDB).filter(
        TenderDocumentDB.tender_id == tender_id
    ).all()

    # Get report attachments (if report exists)
    attachments = []
    if report:
        attachments = db.query(ReportAttachmentDB).filter(
            ReportAttachmentDB.report_id == report.id
        ).all()

    return templates.TemplateResponse("employee_analysis_page.html", {
        "request": request,
        "current_employee": current_employee,
        "tender": tender,
        "assignment": assignment,
        "report": report,
        "is_owner": is_owner,
        "tender_documents": tender_documents,
        "attachments": attachments
    })


@router.post("/api/employee/analysis/{tender_id}/save")
async def save_analysis_report(
    request: Request,
    tender_id: str,
    db: Session = Depends(get_db)
):
    """
    Create or update analysis report.
    Only the report owner can edit.
    """
    current_employee = get_current_employee(request, db)
    if not current_employee:
        return JSONResponse(status_code=401, content={"error": "Authentication required"})

    # Verify employee is assigned to this tender
    assignment = db.query(TenderAssignmentDB).filter(
        and_(
            TenderAssignmentDB.tender_id == tender_id,
            TenderAssignmentDB.employee_id == current_employee.id
        )
    ).first()

    if not assignment:
        return JSONResponse(status_code=403, content={"error": "You are not assigned to this tender"})

    # Parse request body
    try:
        body = await request.json()
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON payload"})

    # Get or create report
    report = db.query(TenderAnalysisReportDB).filter(
        TenderAnalysisReportDB.tender_id == tender_id
    ).first()

    if report:
        # Report exists - verify ownership
        if report.employee_id != current_employee.id:
            return JSONResponse(
                status_code=403,
                content={"error": "Only the report owner can edit this report"}
            )

        # Update existing report
        report.executive_summary = body.get('executive_summary')
        report.eligibility_analysis = body.get('eligibility_analysis')
        report.technical_requirements = body.get('technical_requirements')
        report.financial_assessment = body.get('financial_assessment')
        report.risk_assessment = body.get('risk_assessment')
        report.compliance_review = body.get('compliance_review')
        report.recommendations = body.get('recommendations')
        report.additional_notes = body.get('additional_notes')
        report.status = body.get('status', 'draft')

        # Update version tracking
        report.edit_count += 1
        report.version += 1
        report.last_edited_by = current_employee.id
        report.last_edited_at = datetime.utcnow()
        report.updated_at = datetime.utcnow()

        if body.get('status') == 'submitted' and not report.submitted_at:
            report.submitted_at = datetime.utcnow()

    else:
        # Create new report (unique constraint ensures only one per tender)
        try:
            report = TenderAnalysisReportDB(
                tender_id=tender_id,
                employee_id=current_employee.id,
                executive_summary=body.get('executive_summary'),
                eligibility_analysis=body.get('eligibility_analysis'),
                technical_requirements=body.get('technical_requirements'),
                financial_assessment=body.get('financial_assessment'),
                risk_assessment=body.get('risk_assessment'),
                compliance_review=body.get('compliance_review'),
                recommendations=body.get('recommendations'),
                additional_notes=body.get('additional_notes'),
                status=body.get('status', 'draft'),
                version=1,
                edit_count=0,
                last_edited_by=current_employee.id,
                last_edited_at=datetime.utcnow()
            )

            if body.get('status') == 'submitted':
                report.submitted_at = datetime.utcnow()

            db.add(report)
        except Exception as e:
            db.rollback()
            return JSONResponse(
                status_code=409,
                content={"error": "A report already exists for this tender"}
            )

    try:
        db.commit()
        db.refresh(report)

        return JSONResponse(content={
            "success": True,
            "message": "Report saved successfully",
            "report_id": report.id,
            "status": report.status,
            "version": report.version,
            "edit_count": report.edit_count,
            "last_edited_at": report.last_edited_at.isoformat() if report.last_edited_at else None
        })
    except Exception as e:
        db.rollback()
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/api/employee/analysis/{tender_id}/upload-attachment")
async def upload_attachment(
    request: Request,
    tender_id: str,
    file: UploadFile = File(...),
    description: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Upload file attachment to analysis report.
    Creates report if it doesn't exist.
    """
    current_employee = get_current_employee(request, db)
    if not current_employee:
        return JSONResponse(status_code=401, content={"error": "Authentication required"})

    # Verify employee is assigned to this tender
    assignment = db.query(TenderAssignmentDB).filter(
        and_(
            TenderAssignmentDB.tender_id == tender_id,
            TenderAssignmentDB.employee_id == current_employee.id
        )
    ).first()

    if not assignment:
        return JSONResponse(status_code=403, content={"error": "You are not assigned to this tender"})

    # Validate file
    if file.content_type not in ALLOWED_MIME_TYPES:
        return JSONResponse(
            status_code=400,
            content={"error": f"File type not allowed: {file.content_type}"}
        )

    # Read file data
    file_data = await file.read()
    file_size = len(file_data)

    if file_size > MAX_FILE_SIZE:
        return JSONResponse(
            status_code=400,
            content={"error": f"File too large. Maximum size: {MAX_FILE_SIZE / 1024 / 1024}MB"}
        )

    # Get or create report
    report = db.query(TenderAnalysisReportDB).filter(
        TenderAnalysisReportDB.tender_id == tender_id
    ).first()

    if not report:
        # Create draft report
        report = TenderAnalysisReportDB(
            tender_id=tender_id,
            employee_id=current_employee.id,
            status='draft',
            version=1,
            edit_count=0,
            last_edited_by=current_employee.id,
            last_edited_at=datetime.utcnow()
        )
        db.add(report)
        db.flush()

    # Verify ownership
    if report.employee_id != current_employee.id:
        return JSONResponse(
            status_code=403,
            content={"error": "Only the report owner can upload attachments"}
        )

    # Create attachment
    attachment = ReportAttachmentDB(
        report_id=report.id,
        employee_id=current_employee.id,
        filename=file.filename,
        mime_type=file.content_type,
        file_size=file_size,
        file_data=file_data,
        description=description
    )

    db.add(attachment)

    try:
        db.commit()
        db.refresh(attachment)

        return JSONResponse(content={
            "success": True,
            "message": "File uploaded successfully",
            "attachment": {
                "id": attachment.id,
                "filename": attachment.filename,
                "file_size": attachment.file_size,
                "mime_type": attachment.mime_type,
                "description": attachment.description,
                "uploaded_at": attachment.uploaded_at.isoformat()
            }
        })
    except Exception as e:
        db.rollback()
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.delete("/api/employee/analysis/attachment/{attachment_id}")
async def delete_attachment(
    request: Request,
    attachment_id: int,
    db: Session = Depends(get_db)
):
    """
    Delete file attachment from analysis report.
    Only uploader or report owner can delete.
    """
    current_employee = get_current_employee(request, db)
    if not current_employee:
        return JSONResponse(status_code=401, content={"error": "Authentication required"})

    # Get attachment
    attachment = db.query(ReportAttachmentDB).filter(
        ReportAttachmentDB.id == attachment_id
    ).first()

    if not attachment:
        return JSONResponse(status_code=404, content={"error": "Attachment not found"})

    # Verify ownership (uploader or report owner)
    report = attachment.report
    if attachment.employee_id != current_employee.id and report.employee_id != current_employee.id:
        return JSONResponse(
            status_code=403,
            content={"error": "You don't have permission to delete this attachment"}
        )

    try:
        db.delete(attachment)
        db.commit()

        return JSONResponse(content={
            "success": True,
            "message": "Attachment deleted successfully"
        })
    except Exception as e:
        db.rollback()
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/api/employee/analysis/attachment/{attachment_id}/download")
async def download_attachment(
    request: Request,
    attachment_id: int,
    db: Session = Depends(get_db)
):
    """
    Download file attachment.
    Accessible by assigned employees and users.
    """
    # Check if employee or user
    current_employee = get_current_employee(request, db)
    current_user = None
    if not current_employee:
        current_user = get_current_user(request, db)
        if not current_user:
            raise HTTPException(status_code=401, detail="Authentication required")

    # Get attachment
    attachment = db.query(ReportAttachmentDB).filter(
        ReportAttachmentDB.id == attachment_id
    ).first()

    if not attachment:
        raise HTTPException(status_code=404, detail="Attachment not found")

    # Verify access
    report = attachment.report
    tender_id = report.tender_id

    if current_employee:
        # Verify employee is assigned
        assignment = db.query(TenderAssignmentDB).filter(
            and_(
                TenderAssignmentDB.tender_id == tender_id,
                TenderAssignmentDB.employee_id == current_employee.id
            )
        ).first()
        if not assignment:
            raise HTTPException(status_code=403, detail="Access denied")
    elif current_user:
        # Verify user owns the tender
        tender = db.query(TenderDB).filter(TenderDB.id == tender_id).first()
        # Users can view all analysis reports for tenders they have access to
        # No additional check needed here - users have access to all tenders
        pass

    # Return file
    file_stream = io.BytesIO(attachment.file_data)

    return StreamingResponse(
        file_stream,
        media_type=attachment.mime_type,
        headers={
            'Content-Disposition': f'attachment; filename="{attachment.filename}"',
            'Content-Length': str(attachment.file_size)
        }
    )


@router.get("/api/tender/{tender_id}/analysis-report")
async def get_analysis_report_for_user(
    request: Request,
    tender_id: str,
    db: Session = Depends(get_db)
):
    """
    Get analysis report for tender (user view).
    Returns report with metadata and attachments.
    """
    current_user = get_current_user(request, db)
    if not current_user:
        return JSONResponse(status_code=401, content={"error": "Authentication required"})

    # Get report
    report = db.query(TenderAnalysisReportDB).filter(
        TenderAnalysisReportDB.tender_id == tender_id
    ).first()

    if not report:
        return JSONResponse(status_code=404, content={"error": "No analysis report found for this tender"})

    # Get report owner
    owner = db.query(EmployeeDB).filter(EmployeeDB.id == report.employee_id).first()

    # Get last editor (if different from owner)
    last_editor = None
    if report.last_edited_by and report.last_edited_by != report.employee_id:
        last_editor = db.query(EmployeeDB).filter(EmployeeDB.id == report.last_edited_by).first()

    # Get attachments
    attachments = db.query(ReportAttachmentDB).filter(
        ReportAttachmentDB.report_id == report.id
    ).all()

    attachment_list = []
    for att in attachments:
        uploader = db.query(EmployeeDB).filter(EmployeeDB.id == att.employee_id).first()
        attachment_list.append({
            "id": att.id,
            "filename": att.filename,
            "file_size": att.file_size,
            "mime_type": att.mime_type,
            "description": att.description,
            "uploaded_at": att.uploaded_at.isoformat(),
            "uploader_name": uploader.name if uploader else "Unknown"
        })

    return JSONResponse(content={
        "success": True,
        "report": {
            "id": report.id,
            "status": report.status,
            "owner_name": owner.name if owner else "Unknown",
            "owner_email": owner.email if owner else None,
            "created_at": report.created_at.isoformat(),
            "submitted_at": report.submitted_at.isoformat() if report.submitted_at else None,
            "last_edited_at": report.last_edited_at.isoformat() if report.last_edited_at else None,
            "last_editor_name": last_editor.name if last_editor else (owner.name if owner else None),
            "version": report.version,
            "edit_count": report.edit_count,

            # Report sections
            "executive_summary": report.executive_summary,
            "eligibility_analysis": report.eligibility_analysis,
            "technical_requirements": report.technical_requirements,
            "financial_assessment": report.financial_assessment,
            "risk_assessment": report.risk_assessment,
            "compliance_review": report.compliance_review,
            "recommendations": report.recommendations,
            "additional_notes": report.additional_notes,

            # Attachments
            "attachments": attachment_list
        }
    })
