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
from core.security import user_sessions

# Import font configuration function
import sys
if 'app' in sys.modules:
    from app import get_active_font
else:
    # Fallback if app not loaded yet
    def get_active_font():
        return {"family": "Poppins, sans-serif", "weights": [400, 500, 600, 700]}

"""
=============================================================================
TENDERS API ROUTER - CURRENTLY DISABLED
=============================================================================

This router contains stub implementations that are not yet complete.
The router is commented out in app.py because the stubs return None,
which breaks the working implementations in app.py.

TODO: Implement these 31 stub endpoints to complete the router refactoring:

AUTHENTICATION (4 endpoints):
- POST /api/auth/signup (line 1043)
- POST /api/auth/login (line 1053)
- POST /api/auth/employee/signup (line 1078)
- POST /api/auth/employee/login (line 1088)

COMPANY & PROFILE (3 endpoints):
- POST /api/company-details (line 1133)
- PUT /api/user/profile (line 1611)
- POST /api/company-codes (line 1645)

FAVORITES (2 endpoints):
- POST /api/favorites/{tender_id} (line 1144)
- PUT /api/favorites/{favorite_id} (line 1175)

TENDER SHORTLIST (9 endpoints):
- POST /api/tenders/shortlist (line 1206)
- POST /api/tenders/reject (line 1216)
- POST /api/tenders/shortlist/{id}/progress (line 1249)
- POST /api/tenders/shortlist/{id}/kill (line 1258)
- POST /api/tenders/shortlist/upload-document (line 1270)
- GET /api/tenders/shortlist/{id}/documents (line 1279)
- GET /api/tenders/shortlist/document/{id}/download (line 1288)
- DELETE /api/tenders/shortlist/document/{id} (line 1297)

CUSTOM CARDS (4 endpoints):
- POST /api/custom-cards (line 1501)
- PUT /api/custom-cards/{card_id} (line 1515)
- POST /api/custom-cards/{card_id}/search (line 1588) ⚠️ CRITICAL
- GET /api/custom-cards/{card_id}/tenders (line 1599) ⚠️ CRITICAL

PROJECT (1 endpoint):
- POST /add_project (line 1555)

EMPLOYEE TASKS (7 endpoints):
- POST /api/employee/tasks/{task_id}/comment (line 1781)
- POST /api/employee/messages (line 1791)
- GET /api/employee/messages/{assignment_id} (line 1801)
- GET /api/employee/messages/tender/{tender_id} (line 1811)
- POST /api/employee/assignments (line 1823)
- POST /api/employee/tasks (line 1837)
- POST /api/employee/assignments-with-tasks (line 1850)

For each stub, copy the implementation from app.py and adapt it to work
in this router file. Implementations can be found in app.py starting around
line 13000+.

Once all stubs are implemented and tested, uncomment the router in app.py:
    from api.routes import tenders as tenders_router
    app.include_router(tenders_router.router)

=============================================================================
"""

router = APIRouter()
templates = Jinja2Templates(directory="templates")

@router.get("/", response_class=HTMLResponse)
def home(request: Request, db: Session = Depends(get_db)):
    """Home page with tender listings."""
    # Redirect logged-in users to dashboard
    current_user = get_current_user(request, db)
    if current_user:
        return RedirectResponse(url="/dashboard", status_code=302)

    # Get search and filter parameters
    search = request.query_params.get('search', '')
    category = request.query_params.get('category', '')
    state = request.query_params.get('state', '')
    source = request.query_params.get('source', '')
    min_value = request.query_params.get('min_value', '')
    max_value = request.query_params.get('max_value', '')
    page = int(request.query_params.get('page', 1))
    per_page = 20

    # Check if any filters are applied
    filters_applied = any([search, category, state, source, min_value, max_value])

    tenders = []
    total_tenders = 0
    total_pages = 0
    has_prev = False
    has_next = False

    if filters_applied:
        # Build query only if filters are applied
        query = db.query(TenderDB)

        # Apply filters
        if search:
            search_term = f"%{search}%"
            query = query.filter(
                or_(
                    TenderDB.title.ilike(search_term),
                    TenderDB.authority.ilike(search_term),
                    TenderDB.summary.ilike(search_term)
                )
            )

        if category:
            query = query.filter(TenderDB.category == category)

        if state:
            query = query.filter(TenderDB.state == state)

        if source:
            query = query.filter(TenderDB.source == source)

        if min_value:
            try:
                query = query.filter(TenderDB.estimated_value >= float(min_value))
            except ValueError:
                pass

        if max_value:
            try:
                query = query.filter(TenderDB.estimated_value <= float(max_value))
            except ValueError:
                pass

        # Get total count for pagination
        total_tenders = query.count()

        # Apply pagination and ordering
        tenders = query.order_by(desc(TenderDB.published_at)).offset((page - 1) * per_page).limit(per_page).all()

        # Pagination info
        total_pages = (total_tenders + per_page - 1) // per_page
        has_prev = page > 1
        has_next = page < total_pages

    # Get current user
    current_user = get_current_user(request, db)

    # Get unique categories and states for filters
    categories = db.query(TenderDB.category).distinct().filter(TenderDB.category != '').all()
    categories = [c[0] for c in categories if c[0]]

    states = db.query(TenderDB.state).distinct().filter(TenderDB.state != '').all()
    states = [s[0] for s in states if s[0]]

    sources = db.query(TenderDB.source).distinct().filter(TenderDB.source != '').all()
    sources = [s[0] for s in sources if s[0]]

    return templates.TemplateResponse("home.html", {
        "request": request,
        "tenders": tenders,
        "current_user": current_user,
        "categories": sorted(categories),
        "states": sorted(states),
        "sources": sorted(sources),
        "search": search,
        "category": category,
        "state": state,
        "source": source,
        "min_value": min_value,
        "max_value": max_value,
        "page": page,
        "total_pages": total_pages,
        "has_prev": has_prev,
        "has_next": has_next,
        "total_tenders": total_tenders,
        "selected_font": get_active_font()
    })


@router.get("/procurement", response_class=HTMLResponse)
@require_company_details
async def procurement(request: Request, db: Session = Depends(get_db)):
    """Procurement page with tender listings (login required)."""
    # Check if user is logged in
    current_user = get_current_user(request, db)
    if not current_user:
        return RedirectResponse(url="/login", status_code=302)

    # Get search and filter parameters
    search = request.query_params.get('search', '')
    category = request.query_params.get('category', '')
    state = request.query_params.get('state', '')
    source = request.query_params.get('source', '')
    min_value = request.query_params.get('min_value', '')
    max_value = request.query_params.get('max_value', '')
    page = int(request.query_params.get('page', 1))
    per_page = 20

    # Check if any filters are applied
    filters_applied = any([search, category, state, source, min_value, max_value])

    tenders = []
    total_tenders = 0
    total_pages = 0
    has_prev = False
    has_next = False

    if filters_applied:
        # Build query only if filters are applied
        query = db.query(TenderDB)

        # Apply filters
        if search:
            search_term = f"%{search}%"
            query = query.filter(
                or_(
                    TenderDB.title.ilike(search_term),
                    TenderDB.authority.ilike(search_term),
                    TenderDB.summary.ilike(search_term)
                )
            )

        if category:
            query = query.filter(TenderDB.category == category)

        if state:
            query = query.filter(TenderDB.state == state)

        if source:
            query = query.filter(TenderDB.source == source)

        if min_value:
            try:
                query = query.filter(TenderDB.estimated_value >= float(min_value))
            except ValueError:
                pass

        if max_value:
            try:
                query = query.filter(TenderDB.estimated_value <= float(max_value))
            except ValueError:
                pass

        # Get total count for pagination
        total_tenders = query.count()

        # Apply pagination and ordering
        tenders = query.order_by(desc(TenderDB.published_at)).offset((page - 1) * per_page).limit(per_page).all()

        # Pagination info
        total_pages = (total_tenders + per_page - 1) // per_page
        has_prev = page > 1
        has_next = page < total_pages

    # Get unique categories and states for filters
    categories = db.query(TenderDB.category).distinct().filter(TenderDB.category != '').all()
    categories = [c[0] for c in categories if c[0]]

    states = db.query(TenderDB.state).distinct().filter(TenderDB.state != '').all()
    states = [s[0] for s in states if s[0]]

    sources = db.query(TenderDB.source).distinct().filter(TenderDB.source != '').all()
    sources = [s[0] for s in sources if s[0]]

    return templates.TemplateResponse("tender_searching.html", {
        "request": request,
        "tenders": tenders,
        "current_user": current_user,
        "categories": sorted(categories),
        "states": sorted(states),
        "sources": sorted(sources),
        "search": search,
        "category": category,
        "state": state,
        "source": source,
        "min_value": min_value,
        "max_value": max_value,
        "page": page,
        "total_pages": total_pages,
        "has_prev": has_prev,
        "has_next": has_next,
        "total_tenders": total_tenders,
        "selected_font": get_active_font()
    })


@router.get("/show-all-tneders", response_class=HTMLResponse)
async def show_all_tneders(
    request: Request,
    search: str = "",
    db: Session = Depends(get_db)
):
    """Temporary troubleshooting view that lists every tender without filters."""
    query = db.query(TenderDB)

    if search:
        search_term = f"%{search}%"
        query = query.filter(
            or_(
                TenderDB.title.ilike(search_term),
                TenderDB.summary.ilike(search_term),
                TenderDB.tender_reference_number.ilike(search_term),
                TenderDB.tender_id.ilike(search_term),
                TenderDB.organisation_chain.ilike(search_term),
                TenderDB.authority.ilike(search_term)
            )
        )

    tenders = query.order_by(desc(TenderDB.scraped_at), desc(TenderDB.published_at)).all()

    return templates.TemplateResponse("show_all_tneders.html", {
        "request": request,
        "tenders": tenders,
        "search": search,
        "total_tenders": len(tenders),
        "selected_font": get_active_font()
    })


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Login/signup page."""
    current_user = get_current_user(request, next(get_db()))
    if current_user:
        return RedirectResponse(url="/dashboard", status_code=302)

    # Handle error messages from failed login/signup attempts
    error_message = None
    signup_error_message = None
    error_param = request.query_params.get('error')

    if error_param == 'invalid_credentials':
        error_message = "Invalid email or password. Please check your credentials and try again."
    elif error_param == 'email_exists':
        signup_error_message = "Email already registered. Please use a different email or try logging in."

    return templates.TemplateResponse("login.html", {
        "request": request,
        "error": error_message,
        "signup_error": signup_error_message,
        "selected_font": get_active_font()
    })


@router.get("/employee/login", response_class=HTMLResponse)
async def employee_login_page(request: Request):
    """Employee login/signup page."""
    current_employee = get_current_employee(request, next(get_db()))
    if current_employee:
        return RedirectResponse(url="/employee/dashboard", status_code=302)

    # Handle error messages from failed login/signup attempts
    error_message = None
    signup_error_message = None
    error_param = request.query_params.get('error')

    if error_param == 'invalid_credentials':
        error_message = "Invalid email or password. Please check your credentials and try again."
    elif error_param == 'email_exists':
        signup_error_message = "Email already registered. Please use a different email or try logging in."
    elif error_param == 'invalid_code':
        signup_error_message = "Invalid company code. Please check with your company administrator."

    return templates.TemplateResponse("employee_login.html", {
        "request": request,
        "error": error_message,
        "signup_error": signup_error_message,
        "selected_font": get_active_font()
    })


@router.get("/dashboard", response_class=HTMLResponse)
@require_company_details
async def dashboard(request: Request, db: Session = Depends(get_db)):
    """User dashboard with favorites."""
    current_user = get_current_user(request, db)
    if not current_user:
        return RedirectResponse(url="/login", status_code=302)

    # Get user's favorite tenders with user_filled_data
    favorites = db.query(FavoriteDB).filter(FavoriteDB.user_id == current_user.id).all()
    favorite_tenders = []

    for fav in favorites:
        if fav.tender:
            # Merge tender data with user filled data
            tender_data = {
                'id': fav.tender.id,
                'favorite_id': fav.id,
                'created_at': fav.created_at,
                'status': fav.status,
                'user_filled_data': fav.user_filled_data or {},
                'notes': fav.notes,
                'tender': fav.tender
            }
            favorite_tenders.append(tender_data)

    # Get user's company details
    company_details = db.query(CompanyDB).filter(CompanyDB.user_id == current_user.id).first()

    # Get counts for tender management stats
    favorite_count = len(favorites)
    shortlisted_count = db.query(ShortlistedTenderDB).filter(
        ShortlistedTenderDB.user_id == current_user.id
    ).count()

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "current_user": current_user,
        "company_details": company_details,
        "favorite_tenders": favorite_tenders,
        "favorites": favorites,
        "favorite_count": favorite_count,
        "shortlisted_count": shortlisted_count,
        "now": datetime.utcnow,  # Add current datetime function to template context
        "selected_font": get_active_font()
    })


@router.get("/company-details", response_class=HTMLResponse)
async def company_details_page(request: Request, db: Session = Depends(get_db)):
    """Company details form page."""
    current_user = get_current_user(request, db)
    if not current_user:
        return RedirectResponse(url="/login", status_code=302)

    # Get existing company details if any
    company_details = db.query(CompanyDB).filter(CompanyDB.user_id == current_user.id).first()

    return templates.TemplateResponse("company_details.html", {
        "request": request,
        "current_user": current_user,
        "company_details": company_details,
        "selected_font": get_active_font()
    })


@router.get("/manage_certificates", response_class=HTMLResponse)
@require_company_details
async def manage_certificates_page(request: Request, db: Session = Depends(get_db)):
    """Manage certificates page (upload and search)."""
    current_user = get_current_user(request, db)
    if not current_user:
        return RedirectResponse(url="/login", status_code=302)

    return templates.TemplateResponse("manage_certificates.html", {
        "request": request,
        "current_user": current_user,
        "selected_font": get_active_font()
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
        "current_user": current_user,
        "selected_font": get_active_font()
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
        "certificate": certificate,
        "selected_font": get_active_font()
    })


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
                parsed_sectors = json.loads(company_details.industry_sector)
                # Handle both list of strings and list of dicts
                if parsed_sectors and isinstance(parsed_sectors[0], dict):
                    # Extract just the sector names from dictionaries
                    user_industry_sectors = [s.get('sector', str(s)) for s in parsed_sectors]
                else:
                    user_industry_sectors = parsed_sectors
            except (json.JSONDecodeError, IndexError, AttributeError):
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
        "jv_partners": sorted(jv_partners),
        "sector_analytics": {},  # Empty for now - analytics are loaded via API
        "sector_fee_analytics": {},  # Empty for now - analytics are loaded via API
        "is_test_mode": False,  # Test mode disabled
        "test_token": "",  # No test token
        "test_base_url": "",  # No test base URL
        "selected_font": get_active_font()
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
async def project_detail(request: Request, project_id: int, db: Session = Depends(get_db)):
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
    if project.documents:
        for doc_type, file_paths in project.documents.items():
            documents_with_sizes[doc_type] = []
            for file_path in file_paths:
                try:
                    # Get file size in bytes
                    file_size = os.path.getsize(file_path)
                    documents_with_sizes[doc_type].append({
                        'path': file_path,
                        'size': file_size
                    })
                except (OSError, IOError):
                    # Handle missing or inaccessible files
                    documents_with_sizes[doc_type].append({
                        'path': file_path,
                        'size': None  # Will display as 'Unknown size'
                    })

    return templates.TemplateResponse("project_detail.html", {
        "request": request,
        "current_user": current_user,
        "project": project,
        "documents_with_sizes": documents_with_sizes,
        "selected_font": get_active_font()
    })


@router.get("/api/project/{project_id}/download/{doc_type}/{file_index}")
async def download_project_document(
    project_id: int,
    doc_type: str,
    file_index: int,
    request: Request,
    db: Session = Depends(get_db)):
    """Download a specific document from a project."""
    current_user = get_current_user(request, db)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Get the project
    project = db.query(ProjectDB).filter(
        and_(ProjectDB.id == project_id, ProjectDB.user_id == current_user.id)
    ).first()

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Check if document type exists and has files
    if not project.documents or doc_type not in project.documents:
        raise HTTPException(status_code=404, detail="Document type not found")

    files = project.documents[doc_type]
    if file_index >= len(files):
        raise HTTPException(status_code=404, detail="File not found")

    file_path = files[file_index]

    # Check if file exists
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found on server")

    # Return file
    filename = os.path.basename(file_path)
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/octet-stream'
    )

@router.get("/api/project/{project_id}/download/pdf")
async def download_project_pdf(
    project_id: int,
    request: Request,
    db: Session = Depends(get_db)):
    """Generate and download a PDF summary of the project."""
    current_user = get_current_user(request, db)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Get the project
    project = db.query(ProjectDB).filter(
        and_(ProjectDB.id == project_id, ProjectDB.user_id == current_user.id)
    ).first()

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # For now, return a simple response indicating PDF generation is not yet implemented
    raise HTTPException(status_code=501, detail="PDF generation not yet implemented")

@router.get("/employee/dashboard", response_class=HTMLResponse)
async def employee_dashboard(request: Request, db: Session = Depends(get_db)):
    """Employee dashboard with assigned tenders."""
    current_employee = get_current_employee(request, db)
    if not current_employee:
        return RedirectResponse(url="/employee/login", status_code=302)

    # Get employee's assignments
    assignments = db.query(TenderAssignmentDB).filter(
        TenderAssignmentDB.employee_id == current_employee.id
    ).all()

    # Calculate statistics
    active_tenders = 0
    pending_tasks = 0
    overdue_tasks = 0
    due_today = 0
    now = datetime.utcnow()

    tender_summaries = []
    for assignment in assignments:
        if not assignment.tender:
            continue

        # Check if tender is active
        if assignment.tender.deadline is not None and assignment.tender.deadline > now:
            active_tenders += 1

        # Count tasks for this assignment
        tasks = db.query(TaskDB).filter(TaskDB.assignment_id == assignment.id).all()
        completed_tasks = len([t for t in tasks if t.status == 'completed']) # type: ignore
        total_tasks = len(tasks)
        pending_tasks += total_tasks - completed_tasks

        # Check for overdue tasks
        for task in tasks:
            if task.deadline is not None and task.status != 'completed':  # type: ignore
                if task.deadline < now:  # type: ignore
                    overdue_tasks += 1
                elif task.deadline.date() == now.date():  # type: ignore
                    due_today += 1

        # Find nearest deadline
        nearest_deadline = None
        if tasks:
            active_tasks = [t for t in tasks if t.status != 'completed' and t.deadline is not None]  # type: ignore
            if active_tasks:
                deadlines = [t.deadline for t in active_tasks if t.deadline is not None]
                if deadlines:
                    nearest_deadline = min(deadlines)  # type: ignore

        tender_summaries.append({
            'id': assignment.tender.id,
            'title': assignment.tender.title,
            'tasks_left': total_tasks - completed_tasks,
            'total_tasks': total_tasks,
            'nearest_deadline': nearest_deadline,
            'priority': assignment.priority,
            'role': assignment.role
        })

    return templates.TemplateResponse("employee_dashboard.html", {
        "request": request,
        "current_employee": current_employee,
        "tender_summaries": tender_summaries,
        "active_tenders": active_tenders,
        "pending_tasks": pending_tasks,
        "overdue_tasks": overdue_tasks,
        "due_today": due_today,
        "now": datetime.utcnow(),
        "selected_font": get_active_font()
    })


@router.get("/tender-management", response_class=HTMLResponse)
async def tender_management_page(request: Request, db: Session = Depends(get_db)):
    """Tender management page showing favorites, shortlisted, and awarded tenders."""
    current_user = get_current_user(request, db)
    if not current_user:
        return RedirectResponse(url="/login", status_code=302)

    # Get favorite tenders with tender details
    favorites = db.query(FavoriteDB).join(TenderDB).filter(
        FavoriteDB.user_id == current_user.id
    ).order_by(desc(FavoriteDB.created_at)).all()

    # Get shortlisted tenders with tender details
    shortlisted_tenders = db.query(ShortlistedTenderDB).join(TenderDB).filter(
        ShortlistedTenderDB.user_id == current_user.id
    ).order_by(desc(ShortlistedTenderDB.created_at)).all()

    # Get counts
    favorite_count = len(favorites)
    shortlisted_count = len(shortlisted_tenders)
    awarded_count = db.query(TenderDB).filter(
        and_(TenderDB.awarded == True, TenderDB.awarded_by == current_user.id)
    ).count()

    return templates.TemplateResponse("tender_management.html", {
        "request": request,
        "current_user": current_user,
        "favorites": favorites,
        "shortlisted_tenders": shortlisted_tenders,
        "favorite_count": favorite_count,
        "shortlisted_count": shortlisted_count,
        "awarded_count": awarded_count,
        "selected_font": get_active_font()
    })


@router.get("/employee/task-assignment", response_class=HTMLResponse)
async def employee_task_assignment_page(request: Request, db: Session = Depends(get_db)):
    """Employee task assignment page for managers."""
    current_user = get_current_user(request, db)
    if not current_user:
        return RedirectResponse(url="/login", status_code=302)

    # Get all tenders (for assignment)
    tenders = db.query(TenderDB).filter(TenderDB.awarded == True).order_by(desc(TenderDB.published_at)).limit(50).all()

    # Get user's employees
    company_codes = db.query(CompanyCodeDB).filter(CompanyCodeDB.user_id == current_user.id).all()
    employees = []
    for code in company_codes:
        employees.extend(db.query(EmployeeDB).filter(EmployeeDB.company_code_id == code.id).all())

    return templates.TemplateResponse("work_assignment.html", {
        "request": request,
        "current_user": current_user,
        "tenders": tenders,
        "employees": employees,
        "selected_font": get_active_font()
    })


@router.get("/team/{tender_id}", response_class=HTMLResponse)
@require_company_details
async def team_management_page(request: Request, tender_id: str, db: Session = Depends(get_db)):
    """Team management page for a specific tender."""
    current_user = get_current_user(request, db)
    if not current_user:
        return RedirectResponse(url="/login", status_code=302)

    # Get the tender
    tender = db.query(TenderDB).filter(TenderDB.id == tender_id).first()
    if not tender:
        raise HTTPException(status_code=404, detail="Tender not found")

    # Check if tender is awarded and belongs to user's company
    if tender.awarded != True:
        raise HTTPException(status_code=403, detail="Tender is not awarded")

    # Get user's company codes to filter assignments
    company_codes = db.query(CompanyCodeDB).filter(CompanyCodeDB.user_id == current_user.id).all()
    company_code_ids = [code.id for code in company_codes]

    # Get assignments for this tender where employees belong to user's company
    assignments = db.query(TenderAssignmentDB).join(EmployeeDB).filter(
        and_(TenderAssignmentDB.tender_id == tender_id, EmployeeDB.company_code_id.in_(company_code_ids))
    ).all()

    # Format assignments with tasks
    formatted_assignments = []
    for assignment in assignments:
        if not assignment.employee:
            continue

        # Get tasks for this assignment
        tasks = db.query(TaskDB).filter(TaskDB.assignment_id == assignment.id).all()

        formatted_assignments.append({
            'id': assignment.id,
            'employee': assignment.employee,
            'role': assignment.role,
            'priority': assignment.priority,
            'tasks': tasks
        })

    # Get comments for this tender (from all assignments for this tender)
    assignment_ids = [a.id for a in assignments]
    comments = []
    if assignment_ids:
        comments = db.query(TaskCommentDB).join(TaskDB).filter(
            TaskDB.assignment_id.in_(assignment_ids)
        ).order_by(desc(TaskCommentDB.created_at)).limit(50).all()

    # Get messages for this tender (from all assignments for this tender)
    messages = []
    if assignment_ids:
        messages = db.query(TenderMessageDB).filter(
            TenderMessageDB.assignment_id.in_(assignment_ids)
        ).order_by(TenderMessageDB.created_at.asc()).limit(100).all()

    return templates.TemplateResponse("team.html", {
        "request": request,
        "current_user": current_user,
        "tender": tender,
        "assignments": formatted_assignments,
        "comments": comments,
        "messages": messages,
        "now": datetime.utcnow(),
        "selected_font": get_active_font()
    })


@router.get("/tender/{tender_id}", response_class=HTMLResponse)
@require_company_details
async def tender_detail(request: Request, tender_id: str, db: Session = Depends(get_db)):
    """Tender detail page."""
    tender = db.query(TenderDB).filter(TenderDB.id == tender_id).first()
    if not tender:
        raise HTTPException(status_code=404, detail="Tender not found")

    current_user = get_current_user(request, db)

    # Check if user has favorited this tender
    is_favorited = False
    is_shortlisted = False
    is_rejected = False

    if current_user:
        favorite = db.query(FavoriteDB).filter(
            and_(FavoriteDB.user_id == current_user.id, FavoriteDB.tender_id == tender.id)
        ).first()
        is_favorited = favorite is not None

        # Check if shortlisted
        shortlisted = db.query(ShortlistedTenderDB).filter(
            and_(ShortlistedTenderDB.user_id == current_user.id, ShortlistedTenderDB.tender_id == tender.id)
        ).first()
        is_shortlisted = shortlisted is not None

        # Check if rejected
        rejected = db.query(RejectedTenderDB).filter(
            and_(RejectedTenderDB.user_id == current_user.id, RejectedTenderDB.tender_id == tender.id)
        ).first()
        is_rejected = rejected is not None

    return_to = request.query_params.get('return_to')
    if return_to:
        return_to = unquote(return_to)
        if not return_to.startswith('/'):
            return_to = None

    return templates.TemplateResponse("tender_detail.html", {
        "request": request,
        "tender": tender,
        "current_user": current_user,
        "is_favorited": is_favorited,
        "is_shortlisted": is_shortlisted,
        "is_rejected": is_rejected,
        "return_to": return_to,
        "selected_font": get_active_font(),
    })


@router.get("/custom-card/{card_id}", response_class=HTMLResponse)
@require_company_details
async def custom_card_tenders(request: Request, card_id: int, db: Session = Depends(get_db)):
    """Custom card tenders page."""
    current_user = get_current_user(request, db)
    if not current_user:
        return RedirectResponse(url="/login", status_code=302)

    # Find the card
    card = db.query(CustomCardDB).filter(
        and_(CustomCardDB.id == card_id, CustomCardDB.user_id == current_user.id)
    ).first()

    if not card:
        raise HTTPException(status_code=404, detail="Custom card not found")

    # Get page parameters
    page = int(request.query_params.get('page', 1))
    per_page = 20
    skip = (page - 1) * per_page

    # Get tenders for this card
    result = get_custom_card_tenders(card_id, skip, per_page, request, db)
    tenders = result["tenders"]
    total_tenders = result["total"]

    # Pagination info
    total_pages = (total_tenders + per_page - 1) // per_page
    has_prev = page > 1
    has_next = page < total_pages

    return templates.TemplateResponse("custom_card_tenders.html", {
        "request": request,
        "current_user": current_user,
        "card": result["card"],
        "tenders": tenders,
        "page": page,
        "total_pages": total_pages,
        "has_prev": has_prev,
        "has_next": has_next,
        "total_tenders": total_tenders,
        "selected_font": get_active_font()
    })



# Authentication API endpoints

@router.post("/api/auth/signup")
def signup(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    company: str = Form(""),
    role: str = Form(""),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    """User registration - implementation should be in api/routes/auth.py"""
    pass

@router.post("/api/auth/login")
def login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    """User login - implementation should be in api/routes/auth.py"""
    pass

@router.post("/api/auth/logout")
async def logout(request: Request):
    """User logout."""
    session_token = request.cookies.get('session_token')
    if session_token and session_token in user_sessions:
        del user_sessions[session_token]

    response = RedirectResponse(url="/", status_code=302)
    response.delete_cookie(key="session_token")
    return response

# Employee authentication API endpoints

@router.post("/api/auth/employee/signup")
async def employee_signup(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    company_code: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    """Employee registration - implementation should be in api/routes/auth.py"""
    pass

@router.post("/api/auth/employee/login")
async def employee_login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    """Employee login - implementation should be in api/routes/auth.py"""
    pass

@router.post("/api/auth/employee/logout")
async def employee_logout(request: Request):
    """Employee logout."""
    session_token = request.cookies.get('employee_session_token')
    if session_token and session_token in user_sessions:
        del user_sessions[session_token]

    response = RedirectResponse(url="/", status_code=302)
    response.delete_cookie(key="employee_session_token")
    return response

# Company Details API endpoint

@router.post("/api/company-details")
async def save_company_details(
    request: Request,
    company_name: str = Form(...),
    registration_number: str = Form(...),
    gst_number: str = Form(...),
    pan_number: str = Form(...),
    industry_sector: str = Form(...),
    year_established: int = Form(...),
    annual_turnover: str = Form(...),
    employee_count: str = Form(...),
    registered_address: str = Form(...),
    operational_address: str = Form(""),
    phone_number: str = Form(...),
    email_address: str = Form(...),
    website_url: str = Form(""),
    key_services: str = Form(...),
    specialization_areas: str = Form(...),
    previous_govt_experience: str = Form(""),
    certifications: str = Form(""),
    bank_name: Optional[str] = Form(None),
    account_number: Optional[str] = Form(None),
    ifsc_code: Optional[str] = Form(None),
    account_holder_name: Optional[str] = Form(None),
    managing_director: str = Form(...),
    technical_head: str = Form(""),
    compliance_officer: str = Form(""),
    db: Session = Depends(get_db)
):
    """Save company details - implementation needed"""
    pass

@router.post("/api/favorites/{tender_id}")
@require_company_details
async def add_favorite(
    request: Request, 
    tender_id: str, 
    notes: str = Form(""),
    db: Session = Depends(get_db)
):
    """Add tender to favorites - implementation needed"""
    pass

@router.delete("/api/favorites/{tender_id}")
async def remove_favorite(request: Request, tender_id: str, db: Session = Depends(get_db)):
    """Remove tender from favorites."""
    current_user = get_current_user(request, db)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Find and delete favorite
    favorite = db.query(FavoriteDB).filter(
        and_(FavoriteDB.user_id == current_user.id, FavoriteDB.tender_id == tender_id)
    ).first()

    if not favorite:
        raise HTTPException(status_code=404, detail="Favorite not found")

    db.delete(favorite)
    db.commit()

    return {"message": "Removed from favorites"}


@router.put("/api/favorites/{favorite_id}")
async def update_favorite_data(
    request: Request,
    favorite_id: int,
    user_filled_data: str = Form(...),
    db: Session = Depends(get_db)
):
    """Update favorite data - implementation needed"""
    pass

@router.post("/api/favorites/{favorite_id}/submit")
async def submit_favorite(request: Request, favorite_id: int, db: Session = Depends(get_db)):
    """Submit a favorite (mark as completed)."""
    current_user = get_current_user(request, db)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Find the favorite
    favorite = db.query(FavoriteDB).filter(
        and_(FavoriteDB.id == favorite_id, FavoriteDB.user_id == current_user.id)
    ).first()

    if not favorite:
        raise HTTPException(status_code=404, detail="Favorite not found")

    setattr(favorite, 'status', "submitted")
    db.commit()

    return {"message": "Favorite submitted successfully"}


@router.post("/api/tenders/shortlist")
async def shortlist_tender(
    request: Request,
    tender_id: str = Form(...),
    reason: str = Form(...),
    db: Session = Depends(get_db)
):
    """Shortlist tender - implementation needed"""
    pass

@router.post("/api/tenders/reject")
async def reject_tender(
    request: Request,
    tender_id: str = Form(...),
    reason: str = Form(...),
    db: Session = Depends(get_db)
):
    """Reject tender - implementation needed"""
    pass

@router.get("/api/tenders/shortlist/{shortlist_id}/progress")
async def get_shortlist_progress(request: Request, shortlist_id: int, db: Session = Depends(get_db)):
    """Get progress data for a shortlisted tender."""
    current_user = get_current_user(request, db)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Find the shortlisted tender
    shortlisted = db.query(ShortlistedTenderDB).filter(
        and_(
            ShortlistedTenderDB.id == shortlist_id,
            ShortlistedTenderDB.user_id == current_user.id
        )
    ).first()

    if not shortlisted:
        raise HTTPException(status_code=404, detail="Shortlisted tender not found")

    return {
        "success": True,
        "progress_data": shortlisted.progress_data or {}
    }


@router.post("/api/tenders/shortlist/{shortlist_id}/progress")
async def update_shortlist_progress(
    request: Request,
    shortlist_id: int,
    db: Session = Depends(get_db)
):
    """Update shortlist progress - implementation needed"""
    pass

@router.post("/api/tenders/shortlist/{shortlist_id}/kill")
async def kill_shortlisted_tender(
    request: Request,
    shortlist_id: int,
    db: Session = Depends(get_db)
):
    """Kill shortlisted tender - implementation needed"""
    pass

@router.post("/api/tenders/shortlist/upload-document")
async def upload_stage_document(
    request: Request,
    file: UploadFile = File(...),
    stage: str = Form(...),
    shortlist_id: int = Form(...),
    description: str = Form(""),
    db: Session = Depends(get_db)
):
    """Upload stage document - implementation needed"""
    pass

@router.get("/api/tenders/shortlist/{shortlist_id}/documents")
async def get_stage_documents(
    request: Request,
    shortlist_id: int,
    db: Session = Depends(get_db)
):
    """Get stage documents - implementation needed"""
    pass

@router.get("/api/tenders/shortlist/document/{document_id}/download")
async def download_stage_document(
    request: Request,
    document_id: int,
    db: Session = Depends(get_db)
):
    """Download stage document - implementation needed"""
    pass

@router.delete("/api/tenders/shortlist/document/{document_id}")
async def delete_stage_document(
    request: Request,
    document_id: int,
    db: Session = Depends(get_db)
):
    """Delete stage document - implementation needed"""
    pass

@router.get("/api/favorites/export/csv")
async def export_favorites_csv(request: Request, db: Session = Depends(get_db)):
    """Export all user favorites to CSV."""
    current_user = get_current_user(request, db)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Get all favorites with tender data
    favorites = db.query(FavoriteDB).filter(FavoriteDB.user_id == current_user.id).all()

    # Prepare CSV data
    import csv
    import io
    from fastapi.responses import StreamingResponse

    output = io.StringIO()
    writer = csv.writer(output)

    # Write headers
    headers = [
        "Date of Tender Favourited",
        "Date of Tender Publish",
        "Tender Details",
        "Tender Documents",
        "Tender Portal",
        "Tender ID",
        "Tender Stage",
        "Type of Tender",
        "State",
        "Client",
        "EMD",
        "Est. Consultancy Fee",
        "Pre-Bid Date",
        "Submission Date",
        "PQ Eligibility",
        "JV Eligibility"
    ]
    writer.writerow(headers)

    # Write data rows
    for fav in favorites:
        if not fav.tender:
            continue

        row = []

        # Date of Tender Favourited
        row.append(fav.created_at.strftime('%Y-%m-%d %H:%M:%S') if fav.created_at else '') # type: ignore

        # Date of Tender Publish
        row.append(fav.tender.published_at.strftime('%Y-%m-%d') if fav.tender.published_at else '')

        # Tender Details
        row.append(fav.tender.summary or fav.tender.title or '')

        # Tender Documents
        documents = fav.tender.tender_documents
        if isinstance(documents, dict) and documents:
            doc_links = []
            for key, value in documents.items():
                if isinstance(value, str) and (value.startswith('http') or 'pdf' in value.lower()):
                    doc_links.append(value)
            row.append('; '.join(doc_links))
        else:
            row.append('')

        # Tender Portal
        row.append(fav.tender.source_url or '')

        # Tender ID
        row.append(fav.tender.tender_id or '')

        # Get user filled data if available
        user_data = fav.user_filled_data or {}

        # Tender Stage - Use user data first, then original data
        stage = user_data.get('Tender Stage', '')
        if not stage:
            critical_dates = fav.tender.critical_dates
            if isinstance(critical_dates, dict):
                stage = critical_dates.get('Tender Stage', '')
        row.append(stage)

        # Type of Tender - Use user data first, then original data
        tender_type = user_data.get('Type of Tender', '')
        if not tender_type:
            tender_type = fav.tender.tender_type or ''
        row.append(tender_type)

        # State - Use user data first, then original data
        state = user_data.get('State', '')
        if not state:
            state = fav.tender.state or ''
        row.append(state)

        # Client - Use user data first, then original data
        client = user_data.get('Client', '')
        if not client:
            client = fav.tender.authority or ''
        row.append(client)

        # EMD - Use user data first, then original data
        emd = user_data.get('EMD', '')
        if not emd:
            emd_details = fav.tender.emd_fee_details
            if isinstance(emd_details, dict):
                emd = str(emd_details.get('Amount', ''))
        row.append(emd)

        # Est. Consultancy Fee - Use user data first, then original data
        fee = user_data.get('Est. Consultancy Fee', '')
        if not fee:
            tender_fee_details = fav.tender.tender_fee_details
            if isinstance(tender_fee_details, dict):
                fee = str(tender_fee_details.get('Amount', ''))
        row.append(fee)

        # Pre-Bid Date - Use user data first, then original data
        pre_bid = user_data.get('Pre-Bid Date', '')
        if not pre_bid:
            critical_dates = fav.tender.critical_dates
            if isinstance(critical_dates, dict):
                pre_bid = critical_dates.get('Pre Bid Meeting Date', '')
        row.append(pre_bid)

        # Submission Date - Use user data first, then original data
        submission = user_data.get('Submission Date', '')
        if not submission:
            critical_dates = fav.tender.critical_dates
            if isinstance(critical_dates, dict):
                submission = critical_dates.get('Bid Submission End Date', '')
        row.append(submission)

        # PQ Eligibility - Use user data first, then original data
        pq = user_data.get('PQ Eligibility', '')
        if not pq:
            additional_fields = fav.tender.additional_fields
            if isinstance(additional_fields, dict):
                pq = additional_fields.get('PQ Eligibility', '')
        row.append(pq)

        # JV Eligibility - Use user data first, then original data
        jv = user_data.get('JV Eligibility', '')
        if not jv:
            additional_fields = fav.tender.additional_fields
            if isinstance(additional_fields, dict):
                jv = additional_fields.get('JV Eligibility', '')
        row.append(jv)

        writer.writerow(row)

    output.seek(0)

    # Return CSV file
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=favorites_export.csv"}
    )


@router.get("/api/custom-cards")
@require_company_details
async def get_custom_cards(request: Request, db: Session = Depends(get_db)):
    """Get user's custom cards."""
    current_user = get_current_user(request, db)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    cards = db.query(CustomCardDB).filter(CustomCardDB.user_id == current_user.id).all()
    return {"cards": [
        {
            "id": card.id,
            "card_name": card.card_name,
            "core_search_terms": card.core_search_terms,
            "state": card.state,
            "source": card.source,
            "tender_type": card.tender_type,
            "sector": card.sector,
            "sub_sector": card.sub_sector,
            "work_type": card.work_type,
            "created_at": card.created_at.isoformat() if card.created_at else None, # type: ignore
            "updated_at": card.updated_at.isoformat() if card.updated_at else None # type: ignore
        } for card in cards
    ]}


@router.post("/api/custom-cards")
@require_company_details
async def create_custom_card(
    request: Request,
    card_name: str = Form(...),
    core_search_terms: str = Form(...),
    state: str = Form(""),
    source: str = Form(""),
    tender_type: str = Form(""),
    sector: str = Form(""),
    sub_sector: str = Form(""),
    work_type: str = Form(""),
    db: Session = Depends(get_db)
):
    """Create custom card - implementation needed"""
    pass

@router.put("/api/custom-cards/{card_id}")
async def update_custom_card(
    request: Request,
    card_id: int,
    card_name: str = Form(...),
    core_search_terms: str = Form(...),
    state: str = Form(""),
    source: str = Form(""),
    tender_type: str = Form(""),
    db: Session = Depends(get_db)
):
    """Update custom card - implementation needed"""
    pass

@router.get("/add_project", response_class=HTMLResponse)
@require_company_details
async def add_project_page(request: Request, db: Session = Depends(get_db)):
    """Show the add project form."""
    current_user = get_current_user(request, db)
    if not current_user:
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse("add_project.html", {
        "request": request,
        "current_user": current_user,
        "selected_font": get_active_font()
    })


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
):
    """Submit project - implementation needed"""
    pass

@router.delete("/api/custom-cards/{card_id}")
async def delete_custom_card(request: Request, card_id: int, db: Session = Depends(get_db)):
    """Delete a custom card."""
    current_user = get_current_user(request, db)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Find and delete card
    card = db.query(CustomCardDB).filter(
        and_(CustomCardDB.id == card_id, CustomCardDB.user_id == current_user.id)
    ).first()

    if not card:
        raise HTTPException(status_code=404, detail="Custom card not found")

    db.delete(card)
    db.commit()

    return {"message": "Custom card deleted"}


@router.post("/api/custom-cards/{card_id}/search")
async def search_with_custom_card(
    request: Request,
    card_id: int,
    sector: str = Form(""),
    sub_sector: str = Form(""),
    work_type: str = Form(""),
    db: Session = Depends(get_db)
):
    """Search with custom card - implementation needed"""
    pass

@router.get("/api/custom-cards/{card_id}/tenders")
def get_custom_card_tenders(
    card_id: int,
    skip: int = 0,
    limit: int = 20,
    request: Request = None, # type: ignore
    db: Session = Depends(get_db)
):
    """Get custom card tenders - implementation needed"""
    pass

@router.put("/api/user/profile")
@require_company_details
async def update_profile(
    request: Request,
    name: str = Form(...),
    company: str = Form(""),
    role: str = Form(""),
    db: Session = Depends(get_db)
):
    """Update profile - implementation needed"""
    pass

@router.get("/api/company-codes")
async def get_company_codes(request: Request, db: Session = Depends(get_db)):
    """Get user's company codes."""
    current_user = get_current_user(request, db)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    company_codes = db.query(CompanyCodeDB).filter(CompanyCodeDB.user_id == current_user.id).all()

    # Add employee count for each company code
    codes_with_count = []
    for code in company_codes:
        employee_count = db.query(EmployeeDB).filter(EmployeeDB.company_code_id == code.id).count()
        codes_with_count.append({
            "id": code.id,
            "company_name": code.company_name,
            "company_code": code.company_code,
            "created_at": code.created_at.isoformat(),
            "employee_count": employee_count
        })

    return {"company_codes": codes_with_count}


@router.post("/api/company-codes")
async def create_company_code(
    request: Request,
    company_name: str = Form(...),
    company_code: str = Form(...),
    db: Session = Depends(get_db)
):
    """Create company code - implementation needed"""
    pass

@router.get("/api/tenders")
async def get_tenders(
    skip: int = 0,
    limit: int = 20,
    search: str = "",
    category: str = "",
    state: str = "",
    db: Session = Depends(get_db)
):
    """Get tenders - implementation needed"""
    pass

@router.get("/api/tender/{tender_id}")
async def get_tender(tender_id: str, db: Session = Depends(get_db)):
    """Get single tender details."""
    tender = db.query(TenderDB).filter(TenderDB.id == tender_id).first()
    if not tender:
        raise HTTPException(status_code=404, detail="Tender not found")

    return tender.to_frontend_format()


@router.post("/api/tender/{tender_id}/award")
async def award_tender(request: Request, tender_id: str, db: Session = Depends(get_db)):
    """Mark a tender as awarded for employee task assignment."""
    current_user = get_current_user(request, db)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Find the tender
    tender = db.query(TenderDB).filter(TenderDB.id == tender_id).first()
    if not tender:
        raise HTTPException(status_code=404, detail="Tender not found")

    # Check if tender is already awarded
    if tender.awarded is True:
        raise HTTPException(status_code=400, detail="Tender is already awarded")

    # Mark tender as awarded
    tender.awarded = True  # type: ignore
    tender.awarded_at = datetime.utcnow()  # type: ignore
    tender.awarded_by = current_user.id  # type: ignore

    db.commit()

    return {"message": "Tender marked as awarded successfully", "tender_id": tender_id}


@router.get("/api/tender/{tender_id}/employee-tasks")
async def get_tender_employee_tasks(request: Request, tender_id: str, db: Session = Depends(get_db)):
    """Get employee tasks for a specific tender."""
    current_user = get_current_user(request, db)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Find the tender
    tender = db.query(TenderDB).filter(TenderDB.id == tender_id).first()
    if not tender:
        raise HTTPException(status_code=404, detail="Tender not found")

    # Get all assignments for this tender
    assignments = db.query(TenderAssignmentDB).filter(TenderAssignmentDB.tender_id == tender_id).all()

    employee_tasks = []
    for assignment in assignments:
        if not assignment.employee:
            continue

        # Get tasks for this assignment
        tasks = db.query(TaskDB).filter(TaskDB.assignment_id == assignment.id).all()

        # Format tasks for frontend
        formatted_tasks = []
        for task in tasks:
            formatted_tasks.append({
                'id': task.id,
                'title': task.title,
                'description': task.description,
                'priority': task.priority,
                'status': task.status,
                'deadline': task.deadline.isoformat() if task.deadline else None, # type: ignore
                'created_at': task.created_at.isoformat() if task.created_at else None, # type: ignore
                'completed_at': task.completed_at.isoformat() if task.completed_at else None # type: ignore
            })

        employee_tasks.append({
            'employee_id': assignment.employee.id,
            'employee_name': assignment.employee.name,
            'role': assignment.role,
            'tasks': formatted_tasks
        })

    return {"employee_tasks": employee_tasks}

# Employee Task Management API endpoints

@router.post("/api/employee/tasks/{task_id}/complete")
async def complete_task(request: Request, task_id: int, db: Session = Depends(get_db)):
    """Mark a task as completed."""
    current_employee = get_current_employee(request, db)
    if not current_employee:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Find the task
    task = db.query(TaskDB).filter(TaskDB.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Check if employee is assigned to this task
    assignment = db.query(TenderAssignmentDB).filter(
        and_(TenderAssignmentDB.id == task.assignment_id, TenderAssignmentDB.employee_id == current_employee.id)
    ).first()

    if not assignment:
        raise HTTPException(status_code=403, detail="Not authorized to complete this task")

    # Update task status
    task.status = 'completed'  # type: ignore
    task.completed_at = datetime.utcnow()  # type: ignore
    task.completed_by = current_employee.id  # type: ignore

    db.commit()

    return {"message": "Task marked as completed"}


@router.post("/api/employee/tasks/{task_id}/comment")
async def add_task_comment(
    request: Request,
    task_id: int,
    comment: str = Form(...),
    db: Session = Depends(get_db)
):
    """Add task comment - implementation needed"""
    pass

@router.post("/api/employee/messages")
async def send_message(
    request: Request,
    assignment_id: int = Form(...),
    message: str = Form(...),
    db: Session = Depends(get_db)
):
    """Send message - implementation needed"""
    pass

@router.get("/api/employee/messages/{assignment_id}")
async def get_messages(
    request: Request,
    assignment_id: int,
    since: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get messages - implementation needed"""
    pass

@router.get("/api/employee/messages/tender/{tender_id}")
async def get_tender_messages(
    request: Request,
    tender_id: str,
    since: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get tender messages - implementation needed"""
    pass

@router.post("/api/employee/assignments")
async def create_assignment(
    request: Request,
    tender_id: str = Form(...),
    employee_id: str = Form(...),
    role: str = Form(...),
    priority: str = Form("medium"),
    db: Session = Depends(get_db)
):
    """Create assignment - implementation needed"""
    pass

@router.post("/api/employee/tasks")
async def create_task(
    request: Request,
    assignment_id: int = Form(...),
    title: str = Form(...),
    description: str = Form(""),
    priority: str = Form("medium"),
    estimated_hours: Optional[float] = Form(None),
    deadline: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """Create task - implementation needed"""
    pass

@router.post("/api/employee/assignments-with-tasks")
async def create_assignment_with_tasks(
    request: Request,
    tender_id: str = Form(...),
    employee_id: str = Form(...),
    role: str = Form(...),
    priority: str = Form("medium"),
    tasks: str = Form(...),
    db: Session = Depends(get_db)
):
    """Create assignment with tasks - implementation needed"""
    pass

@router.get("/api/tender/{tender_id}/assignments")
async def get_tender_assignments(request: Request, tender_id: str, db: Session = Depends(get_db)):
    """Get all employee assignments for a specific tender."""
    current_user = get_current_user(request, db)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Check if tender exists
    tender = db.query(TenderDB).filter(TenderDB.id == tender_id).first()
    if not tender:
        raise HTTPException(status_code=404, detail="Tender not found")

    # Get user's company codes to filter assignments
    company_codes = db.query(CompanyCodeDB).filter(CompanyCodeDB.user_id == current_user.id).all()
    company_code_ids = [code.id for code in company_codes]

    # Get assignments for this tender where employees belong to user's company
    assignments = db.query(TenderAssignmentDB).join(EmployeeDB).filter(
        and_(TenderAssignmentDB.tender_id == tender_id, EmployeeDB.company_code_id.in_(company_code_ids))
    ).all()

    # Format assignments for frontend
    formatted_assignments = []
    for assignment in assignments:
        if not assignment.employee:
            continue

        # Get task count for this assignment
        task_count = db.query(TaskDB).filter(TaskDB.assignment_id == assignment.id).count()
        completed_tasks = db.query(TaskDB).filter(
            and_(TaskDB.assignment_id == assignment.id, TaskDB.status == 'completed')
        ).count()

        formatted_assignments.append({
            'id': assignment.id,
            'employee_id': assignment.employee.id,
            'employee_name': assignment.employee.name,
            'employee_email': assignment.employee.email,
            'role': assignment.role,
            'priority': assignment.priority,
            'assigned_at': assignment.assigned_at.isoformat() if assignment.assigned_at else None, # type: ignore
            'task_count': task_count,
            'completed_tasks': completed_tasks
        })

    return {"assignments": formatted_assignments}


@router.delete("/api/employee/assignments/{assignment_id}")
async def unassign_employee(request: Request, assignment_id: int, db: Session = Depends(get_db)):
    """Un-assign an employee from a tender."""
    current_user = get_current_user(request, db)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Find the assignment
    assignment = db.query(TenderAssignmentDB).filter(TenderAssignmentDB.id == assignment_id).first()
    if not assignment:
        raise HTTPException(status_code=404, detail="Assignment not found")

    # Check if assignment belongs to user's company
    company_codes = db.query(CompanyCodeDB).filter(CompanyCodeDB.user_id == current_user.id).all()
    employee_company_codes = [code.id for code in company_codes]
    if assignment.employee.company_code_id not in employee_company_codes:
        raise HTTPException(status_code=403, detail="Assignment does not belong to your company")

    # Delete the assignment (this will cascade delete tasks and messages)
    db.delete(assignment)
    db.commit()

    return {"message": "Employee un-assigned successfully"}


# ============================================================================
# FAVORITES LIST FOR CERTIFICATE ATTACHMENT
# ============================================================================
# MOVED TO api/routes/certificates.py
# This endpoint is needed for certificate attachment and has been moved to the
# active certificates_router. Original implementation was at lines 1994-2052.
# The endpoint is now available in certificates.py around line 3456.
# ============================================================================

@router.get("/api/favorites/list-for-attachment")
async def get_favorites_for_attachment(
    request: Request,
    db: Session = Depends(get_db)
):
    """
    MOVED: This endpoint has been moved to api/routes/certificates.py

    This stub remains here for documentation purposes.
    The working implementation is in the certificates router.
    """
    pass


# Certificate API endpoints

@router.post("/api/certificates/upload")
@require_company_details
async def upload_certificate(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload certificate - implementation needed"""
    pass

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
    """Get certificates - implementation needed"""
    pass

@router.post("/api/certificates/search")
@require_company_details
async def search_certificates_post(
    request: Request,
    db: Session = Depends(get_db)
):
    """Search certificates - implementation needed"""
    pass

@router.get("/api/certificates/{certificate_id}")
async def get_certificate(
    request: Request,
    certificate_id: str,
    db: Session = Depends(get_db)
):
    """Get certificate - implementation needed"""
    pass

@router.get("/api/certificates/{certificate_id}/file")
async def get_certificate_file(
    request: Request,
    certificate_id: str,
    db: Session = Depends(get_db)
):
    """Get certificate file - implementation needed"""
    pass

@router.get("/api/certificates/{certificate_id}/download")
async def download_certificate_file(
    request: Request,
    certificate_id: str,
    db: Session = Depends(get_db)
):
    """Download certificate file - implementation needed"""
    pass

@router.get("/api/certificates/analytics")
async def get_certificate_analytics(
    request: Request,
    db: Session = Depends(get_db)
):
    """Get certificate analytics - implementation needed"""
    pass

@router.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

# Cleanup task (run periodically)

@router.get("/api/admin/cleanup")
async def cleanup_old_data(db: Session = Depends(get_db)):
    """Cleanup old tenders (admin endpoint)."""
    count = cleanup_old_tenders(db, days_old=60)
    return {"message": f"Cleaned up {count} old tenders"}
