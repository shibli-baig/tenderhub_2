"""
Authentication routes for user and employee login/signup.
"""

import uuid
import json
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Request, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from database import (
    UserDB, EmployeeDB, CompanyCodeDB, CompanyDB, FavoriteDB, ShortlistedTenderDB
)
from core.dependencies import get_db, get_current_user, get_current_employee, user_has_complete_company_details
from core.security import hash_password, verify_password, create_session, delete_session

router = APIRouter()
templates = Jinja2Templates(directory="templates")


# Login Pages
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
        "signup_error": signup_error_message
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
        "signup_error": signup_error_message
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
        "company_details": company_details
    })


# User Authentication API endpoints
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
    """User registration."""
    # Check if user already exists
    existing_user = db.query(UserDB).filter(UserDB.email == email).first()
    if existing_user:
        # Redirect to login page with signup error message
        return RedirectResponse(url=f"/login?error=email_exists&email={email}", status_code=302)

    # Create new user
    user_id = str(uuid.uuid4())
    hashed_password = hash_password(password)

    new_user = UserDB(
        id=user_id,
        email=email,
        name=name,
        company=company,
        role=role,
        password_hash=hashed_password,
        created_at=datetime.utcnow()
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    # Create session
    session_token = create_session(user_id)

    # Create response with redirect to company details
    response = RedirectResponse(url="/company-details", status_code=302)
    response.set_cookie(
        key="session_token",
        value=session_token,
        httponly=True,
        max_age=7*24*60*60  # 7 days
    )

    return response


@router.post("/api/auth/login")
def login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    """User login."""
    # Find user
    user = db.query(UserDB).filter(UserDB.email == email).first()
    if not user or not verify_password(password, user.password_hash): # type: ignore
        # Redirect to login page with error message
        return RedirectResponse(url=f"/login?error=invalid_credentials&email={email}", status_code=302)

    # Update last login
    user.last_login = datetime.utcnow() # type: ignore
    db.commit()

    # Create session
    session_token = create_session(user.id) # type: ignore

    # Check if user has complete company details
    redirect_url = "/dashboard" if user_has_complete_company_details(user.id, db) else "/company-details" # type: ignore

    # Create response with redirect
    response = RedirectResponse(url=redirect_url, status_code=302)
    response.set_cookie(
        key="session_token",
        value=session_token,
        httponly=True,
        max_age=7*24*60*60  # 7 days
    )

    return response


@router.post("/api/auth/logout")
async def logout(request: Request):
    """User logout."""
    session_token = request.cookies.get('session_token')
    delete_session(session_token)

    response = RedirectResponse(url="/", status_code=302)
    response.delete_cookie(key="session_token")
    return response


# Employee Authentication API endpoints
@router.post("/api/auth/employee/signup")
async def employee_signup(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    company_code: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    """Employee registration."""
    # Check if employee already exists
    existing_employee = db.query(EmployeeDB).filter(EmployeeDB.email == email).first()
    if existing_employee:
        return RedirectResponse(url=f"/employee/login?error=email_exists&email={email}", status_code=302)

    # Validate company code
    company_code_upper = company_code.upper()
    company_code_record = db.query(CompanyCodeDB).filter(CompanyCodeDB.company_code == company_code_upper).first()
    if not company_code_record:
        return RedirectResponse(url=f"/employee/login?error=invalid_code&email={email}&company_code={company_code}", status_code=302)

    # Create new employee
    employee_id = str(uuid.uuid4())
    hashed_password = hash_password(password)

    new_employee = EmployeeDB(
        id=employee_id,
        email=email,
        name=name,
        company_code_id=company_code_record.id,
        password_hash=hashed_password,
        created_at=datetime.utcnow()
    )

    db.add(new_employee)
    db.commit()
    db.refresh(new_employee)

    # Create session
    session_token = create_session(employee_id)

    # Create response with redirect
    response = RedirectResponse(url="/employee/dashboard", status_code=302)
    response.set_cookie(
        key="employee_session_token",
        value=session_token,
        httponly=True,
        max_age=7*24*60*60  # 7 days
    )

    return response


@router.post("/api/auth/employee/login")
async def employee_login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    """Employee login."""
    # Find employee
    employee = db.query(EmployeeDB).filter(EmployeeDB.email == email).first()
    if not employee or not verify_password(password, employee.password_hash): # type: ignore
        return RedirectResponse(url=f"/employee/login?error=invalid_credentials&email={email}", status_code=302)

    # Update last login
    employee.last_login = datetime.utcnow() # type: ignore
    db.commit()

    # Create session
    session_token = create_session(employee.id) # type: ignore

    # Create response with redirect
    response = RedirectResponse(url="/employee/dashboard", status_code=302)
    response.set_cookie(
        key="employee_session_token",
        value=session_token,
        httponly=True,
        max_age=7*24*60*60  # 7 days
    )

    return response


@router.post("/api/auth/employee/logout")
async def employee_logout(request: Request):
    """Employee logout."""
    session_token = request.cookies.get('employee_session_token')
    delete_session(session_token)

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
    """Save or update company details."""
    current_user = get_current_user(request, db)
    if not current_user:
        return JSONResponse(
            status_code=401,
            content={"error": "Authentication required"}
        )

    # Check if company details already exist
    existing_company = db.query(CompanyDB).filter(CompanyDB.user_id == current_user.id).first()

    # Validate industry_sector JSON format
    try:
        if industry_sector.startswith('[') and industry_sector.endswith(']'):
            # It's a JSON array, validate it
            json.loads(industry_sector)
    except (json.JSONDecodeError, AttributeError):
        # If it's not valid JSON, treat as single value and convert to array
        industry_sector = json.dumps([industry_sector]) if industry_sector else json.dumps([])

    if existing_company:
        # Update existing company details
        existing_company.company_name = company_name
        existing_company.registration_number = registration_number
        existing_company.gst_number = gst_number
        existing_company.pan_number = pan_number
        existing_company.industry_sector = industry_sector
        existing_company.year_established = year_established
        existing_company.annual_turnover = annual_turnover
        existing_company.employee_count = employee_count
        existing_company.registered_address = registered_address
        existing_company.operational_address = operational_address
        existing_company.phone_number = phone_number
        existing_company.email_address = email_address
        existing_company.website_url = website_url
        existing_company.key_services = key_services
        existing_company.specialization_areas = specialization_areas
        existing_company.previous_govt_experience = previous_govt_experience
        existing_company.certifications = certifications
        if bank_name is not None:
            existing_company.bank_name = bank_name
        if account_number is not None:
            existing_company.account_number = account_number
        if ifsc_code is not None:
            existing_company.ifsc_code = ifsc_code
        if account_holder_name is not None:
            existing_company.account_holder_name = account_holder_name
        existing_company.managing_director = managing_director
        existing_company.technical_head = technical_head
        existing_company.compliance_officer = compliance_officer
        existing_company.is_complete = True
        existing_company.updated_at = datetime.utcnow()
        company = existing_company
    else:
        # Create new company details
        company = CompanyDB(
            user_id=current_user.id,
            company_name=company_name,
            registration_number=registration_number,
            gst_number=gst_number,
            pan_number=pan_number,
            industry_sector=industry_sector,
            year_established=year_established,
            annual_turnover=annual_turnover,
            employee_count=employee_count,
            registered_address=registered_address,
            operational_address=operational_address,
            phone_number=phone_number,
            email_address=email_address,
            website_url=website_url,
            key_services=key_services,
            specialization_areas=specialization_areas,
            previous_govt_experience=previous_govt_experience,
            certifications=certifications,
            bank_name=bank_name or '',
            account_number=account_number or '',
            ifsc_code=ifsc_code or '',
            account_holder_name=account_holder_name or '',
            managing_director=managing_director,
            technical_head=technical_head,
            compliance_officer=compliance_officer,
            is_complete=True,
            created_at=datetime.utcnow()
        )
        db.add(company)

    db.commit()
    db.refresh(company)

    return JSONResponse(
        status_code=200,
        content={"message": "Company details saved successfully", "redirect": "/dashboard"}
    )


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, db: Session = Depends(get_db)):
    """User dashboard with favorites."""
    current_user = get_current_user(request, db)
    if not current_user:
        return RedirectResponse(url="/login", status_code=302)

    # Check company details
    if not user_has_complete_company_details(current_user.id, db):
        return RedirectResponse(url="/company-details", status_code=302)

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
        "now": datetime.utcnow  # Add current datetime function to template context
    })
