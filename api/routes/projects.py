from fastapi import APIRouter, Request, Form, Depends, HTTPException, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import json
import uuid
import os
from pathlib import Path
from urllib.parse import unquote

from database import *
from core.dependencies import get_db, get_current_user, get_current_employee, require_company_details


router = APIRouter()
templates = Jinja2Templates(directory="templates")

@router.get("/projects", response_class=HTMLResponse)
@require_company_details
async def projects_list(request: Request, db: Session = Depends(get_db)):
    """List all user projects with search functionality."""
    current_user = get_current_user(request, db)
    if not current_user:
        return RedirectResponse(url="/login", status_code=302)

    # Get user's company details to retrieve industry sectors
    company_details = db.query(CompanyDB).filter(CompanyDB.user_id == current_user.id).first()
    user_industry_sectors = []
    if company_details and company_details.industry_sector:
        if company_details.industry_sector.startswith('['):
            try:
                user_industry_sectors = json.loads(company_details.industry_sector)
            except json.JSONDecodeError:
                user_industry_sectors = [company_details.industry_sector]
        else:
            user_industry_sectors = [company_details.industry_sector]

    # Get search and filter parameters
    search = request.query_params.get('search', '')
    sector_filter = request.query_params.get('sector_filter', '')
    project_name = request.query_params.get('project_name', '')
    client = request.query_params.get('client', '')
    sub_sector = request.query_params.get('sub_sector', '')
    start_date = request.query_params.get('start_date', '')
    end_date = request.query_params.get('end_date', '')
    country = request.query_params.get('country', '')
    jv_partner = request.query_params.get('jv_partner', '')
    page = int(request.query_params.get('page', 1))
    per_page = 10

    # Get total project count for cards
    total_project_count = db.query(ProjectDB).filter(ProjectDB.user_id == current_user.id).count()

    # Only show projects if search parameters are provided
    projects = []
    total_projects = 0
    total_pages = 0
    has_prev = False
    has_next = False

    # Check if any search/filter parameter is provided
    search_performed = any([search, sector_filter, project_name, client, sub_sector, start_date, end_date, country, jv_partner])

    if search_performed:
        # Build query
        query = db.query(ProjectDB).filter(ProjectDB.user_id == current_user.id)

        # Apply filters
        if search:
            search_term = f"%{search}%"
            query = query.filter(
                or_(
                    ProjectDB.project_name.ilike(search_term),
                    ProjectDB.project_description.ilike(search_term),
                    ProjectDB.client_name.ilike(search_term)
                )
            )

        if project_name:
            query = query.filter(ProjectDB.project_name.ilike(f"%{project_name}%"))

        if sector_filter:
            query = query.filter(ProjectDB.sector == sector_filter)

        if client:
            query = query.filter(ProjectDB.client_name.ilike(f"%{client}%"))

        if sub_sector:
            query = query.filter(ProjectDB.sub_sector.ilike(f"%{sub_sector}%"))

        if start_date:
            try:
                start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
                query = query.filter(ProjectDB.start_date >= start_date_obj)
            except ValueError:
                pass

        if end_date:
            try:
                end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
                query = query.filter(ProjectDB.end_date <= end_date_obj)
            except ValueError:
                pass

        if country:
            query = query.filter(ProjectDB.country.ilike(f"%{country}%"))

        if jv_partner:
            query = query.filter(ProjectDB.jv_partner.ilike(f"%{jv_partner}%"))

        # Get total count for pagination
        total_projects = query.count()

        # Apply pagination and ordering
        projects = query.order_by(desc(ProjectDB.created_at)).offset((page - 1) * per_page).limit(per_page).all()

        # Pagination info
        total_pages = (total_projects + per_page - 1) // per_page
        has_prev = page > 1
        has_next = page < total_pages

    # Get unique values for search dropdowns from all projects
    all_projects = db.query(ProjectDB).filter(ProjectDB.user_id == current_user.id).all()
    sectors = list(set([p.sector for p in all_projects if p.sector]))
    clients = list(set([p.client_name for p in all_projects if p.client_name]))
    sub_sectors = list(set([p.sub_sector for p in all_projects if p.sub_sector]))
    countries = list(set([p.country for p in all_projects if p.country]))
    jv_partners = list(set([p.jv_partner for p in all_projects if p.jv_partner]))

    return templates.TemplateResponse("past_projects.html", {
        "request": request,
        "current_user": current_user,
        "projects": projects,
        "search": search,
        "sector_filter": sector_filter,
        "project_name": project_name,
        "client": client,
        "sub_sector": sub_sector,
        "start_date": start_date,
        "end_date": end_date,
        "country": country,
        "jv_partner": jv_partner,
        "page": page,
        "total_pages": total_pages,
        "has_prev": has_prev,
        "has_next": has_next,
        "total_projects": total_projects,
        "total_project_count": total_project_count,
        "search_performed": search_performed,
        "user_industry_sectors": user_industry_sectors,
        "sectors": sorted(sectors),
        "sub_sectors": sorted(sub_sectors),
        "clients": sorted(clients),
        "countries": sorted(countries),
        "jv_partners": sorted(jv_partners)
    })


@router.get("/api/projects/sector-count")
@require_company_details
async def get_projects_sector_count(request: Request, sector: str, db: Session = Depends(get_db)):
    """Get count of projects for a specific sector."""
    current_user = get_current_user(request, db)
    if not current_user:
        return {"error": "User not found", "count": 0}

    count = db.query(ProjectDB).filter(
        ProjectDB.user_id == current_user.id,
        ProjectDB.sector == sector
    ).count()

    return {"count": count}


@router.get("/project/{project_id}", response_class=HTMLResponse)
@require_company_details
async def project_detail(
    request: Request, 
    project_id: int, 
    return_url: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Project detail page with document download options."""
    current_user = get_current_user(request, db)
    if not current_user:
        return RedirectResponse(url="/login", status_code=302)

    # Get the project
    project = db.query(ProjectDB).filter(
        and_(ProjectDB.id == project_id, ProjectDB.user_id == current_user.id)
    ).first()

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Calculate file sizes for all documents
    documents_with_sizes = {}
    if project.documents and isinstance(project.documents, dict):
        for doc_type, file_entries in project.documents.items():
            documents_with_sizes[doc_type] = []
            # Ensure file_entries is a list
            if not isinstance(file_entries, list):
                file_entries = [file_entries] if file_entries else []

            for file_entry in file_entries:
                # Handle two formats:
                # 1. Old format: simple string file path
                # 2. New format: dict with file_path, metadata (for task deliverables)
                if isinstance(file_entry, dict):
                    # New format - deliverable with metadata
                    file_path = file_entry.get("file_path", "")
                    original_filename = file_entry.get("original_filename", os.path.basename(file_path))
                    uploaded_by = file_entry.get("uploaded_by")
                    uploaded_at = file_entry.get("uploaded_at")
                    task_title = file_entry.get("task_title")
                    task_file_id = file_entry.get("task_file_id")
                    stored_size = file_entry.get("file_size")
                    description = file_entry.get("description")
                    
                    try:
                        # Get file size from disk or use stored size
                        if os.path.exists(file_path):
                            file_size = os.path.getsize(file_path)
                        else:
                            file_size = stored_size
                        
                        documents_with_sizes[doc_type].append({
                            'path': file_path,
                            'size': file_size,
                            'original_filename': original_filename,
                            'uploaded_by': uploaded_by,
                            'uploaded_at': uploaded_at,
                            'task_title': task_title,
                            'task_file_id': task_file_id,
                            'description': description,
                            'is_deliverable': True
                        })
                    except Exception as e:
                        documents_with_sizes[doc_type].append({
                            'path': file_path,
                            'size': stored_size,
                            'original_filename': original_filename,
                            'uploaded_by': uploaded_by,
                            'uploaded_at': uploaded_at,
                            'task_title': task_title,
                            'description': description,
                            'is_deliverable': True
                        })
                else:
                    # Old format - simple file path string
                    file_path = file_entry
                    try:
                        # Get file size in bytes
                        file_size = os.path.getsize(file_path)
                        documents_with_sizes[doc_type].append({
                            'path': file_path,
                            'size': file_size,
                            'original_filename': os.path.basename(file_path),
                            'is_deliverable': False
                        })
                    except (OSError, IOError):
                        # Handle missing or inaccessible files
                        documents_with_sizes[doc_type].append({
                            'path': file_path,
                            'size': None,  # Will display as 'Unknown size'
                            'original_filename': os.path.basename(file_path),
                            'is_deliverable': False
                        })

    # Load milestones for timeline
    milestones = db.query(ProjectMilestoneDB).filter(
        ProjectMilestoneDB.project_id == project_id
    ).order_by(ProjectMilestoneDB.milestone_date, ProjectMilestoneDB.display_order).all()
    
    # Ensure default milestones exist if none are present
    if len(milestones) == 0:
        from database import ensure_default_project_milestones
        ensure_default_project_milestones(project, db)
        # Reload milestones after creation
        milestones = db.query(ProjectMilestoneDB).filter(
            ProjectMilestoneDB.project_id == project_id
        ).order_by(ProjectMilestoneDB.milestone_date, ProjectMilestoneDB.display_order).all()

    # Use return_url if provided, otherwise fallback to default
    if return_url:
        from urllib.parse import unquote
        decoded_return_url = unquote(return_url)
        back_url = decoded_return_url
    else:
        back_url = "/projects"
    
    return templates.TemplateResponse("project_detail.html", {
        "request": request,
        "current_user": current_user,
        "project": project,
        "documents_with_sizes": documents_with_sizes,
        "milestones": milestones,
        "test_redirect_url": back_url
    })


@router.get("/api/project/{project_id}/download/{doc_type}/{file_index}")
async def download_project_document(
    project_id: int,
    doc_type: str,
    file_index: int,
    request: Request,
    db: Session = Depends(get_db)

@router.get("/api/project/{project_id}/download/pdf")
async def download_project_pdf(
    project_id: int,
    request: Request,
    db: Session = Depends(get_db)

@router.get("/add_project", response_class=HTMLResponse)
@require_company_details
async def add_project_page(request: Request, db: Session = Depends(get_db)):
    """Show the add project form."""
    current_user = get_current_user(request, db)
    if not current_user:
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse("add_project.html", {"request": request, "current_user": current_user})


@router.post("/add_project")
@require_company_details
async def submit_project(
    request: Request,
    # Section 1: Project Details
    project_name: str = Form(...),
    project_description: str = Form(""),
    client_name: str = Form(""),
    sector: str = Form(""),
    sub_sector: str = Form(""),
    consultancy_fee: Optional[float] = Form(None),
    project_cost: Optional[float] = Form(None),
    start_date: Optional[str] = Form(None),
    end_date: Optional[str] = Form(None),
    ongoing: str = Form("false"),
    jv_partner: str = Form(""),
    country: str = Form("India"),
    states: str = Form(""),
    cities: str = Form(""),
    # Section 2: Services Rendered (will be parsed from form data)
    # Section 3: Documents (file uploads)
    db: Session = Depends(get_db)

