"""
Shared dependencies for FastAPI routes.
"""

from typing import Optional
from functools import wraps
from fastapi import Request, Depends
from fastapi.responses import RedirectResponse
from sqlalchemy import and_
from sqlalchemy.orm import Session

from database import SessionLocal, UserDB, EmployeeDB, CompanyDB, ExpertDB, ExpertProfileDB
from core.security import get_session, get_expert_session


def get_db():
    """Database session dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user(request: Request, db: Session = Depends(get_db)) -> Optional[UserDB]:
    """Get current user from session (supports test sessions)."""
    session_token = request.cookies.get('session_token')
    if not session_token:
        return None

    session_data = get_session(session_token)
    if not session_data:
        return None

    user = db.query(UserDB).filter(UserDB.id == session_data['user_id']).first()

    # Attach test session info to user object for API use
    if user and session_data.get('is_test_session'):
        user._is_test_session = True
        user._restricted_navigation = session_data.get('restricted_navigation', False)
        user._test_token = session_data.get('test_token')
        user._test_base_url = session_data.get('test_base_url')
        user._quarantined = session_data.get('quarantined', False)

    return user


def get_current_employee(request: Request, db: Session = Depends(get_db)) -> Optional[EmployeeDB]:
    """Get current employee from session."""
    session_token = request.cookies.get('employee_session_token')
    if not session_token:
        return None

    session_data = get_session(session_token)
    if not session_data:
        return None

    employee = db.query(EmployeeDB).filter(EmployeeDB.id == session_data['user_id']).first()
    return employee


def get_current_bd_employee(request: Request, db: Session = Depends(get_db)) -> Optional[EmployeeDB]:
    """Get current BD employee from session. Returns employee only if is_bd is True."""
    employee = get_current_employee(request, db)
    if employee and employee.is_bd:
        return employee
    return None


def get_current_user_or_bd_employee(request: Request, db: Session = Depends(get_db)):
    """
    Get either current user or BD employee from session.
    Returns a tuple: (entity, entity_type) where entity_type is 'user' or 'bd_employee'
    """
    # First check for user session
    user = get_current_user(request, db)
    if user:
        return user, 'user'

    # Then check for BD employee session
    bd_employee = get_current_bd_employee(request, db)
    if bd_employee:
        return bd_employee, 'bd_employee'

    return None, None


def get_user_id_for_queries(request: Request, db: Session):
    """
    Get user_id for database queries - works for both regular users and BD employees.
    For BD employees, returns the company owner's user_id.
    Returns (user_id, entity) tuple where entity is the user or BD employee object.
    """
    from database import CompanyCodeDB

    entity, entity_type = get_current_user_or_bd_employee(request, db)
    if not entity:
        return None, None

    if entity_type == 'user':
        return entity.id, entity
    else:  # BD employee
        company_code = db.query(CompanyCodeDB).filter(CompanyCodeDB.id == entity.company_code_id).first()
        if company_code:
            return company_code.user_id, entity
        return None, None


def get_id_for_custom_cards(request: Request, db: Session):
    """
    Get ID for custom cards - BD employees use their own ID, not company owner's.
    This ensures BD employees have separate cards from admin.
    Returns (id, entity, entity_type) tuple.
    """
    entity, entity_type = get_current_user_or_bd_employee(request, db)
    if not entity:
        return None, None, None

    # Both users and BD employees use their own ID for cards
    return entity.id, entity, entity_type


def get_id_for_tender_management(request: Request, db: Session):
    """
    Get ID for tender management (favorites, shortlists, awarded).
    BD employees use their own ID for isolated tender management.
    Returns (id, entity, entity_type) tuple.

    This is identical to get_id_for_custom_cards() - both provide isolation.
    """
    entity, entity_type = get_current_user_or_bd_employee(request, db)
    if not entity:
        return None, None, None

    # Both users and BD employees use their own ID
    return entity.id, entity, entity_type


def get_all_bd_employee_ids_for_company(user_id: str, db: Session):
    """
    For admin users, get all BD employee IDs in their company.
    Used to fetch all BD employee tenders for admin oversight.

    Returns: List of employee IDs (strings)
    """
    from database import CompanyCodeDB, EmployeeDB

    # Get all company codes owned by this user
    company_codes = db.query(CompanyCodeDB).filter(
        CompanyCodeDB.user_id == user_id
    ).all()

    if not company_codes:
        return []

    # Get all BD employees under these company codes
    company_code_ids = [cc.id for cc in company_codes]
    bd_employees = db.query(EmployeeDB).filter(
        and_(
            EmployeeDB.company_code_id.in_(company_code_ids),
            EmployeeDB.is_bd == True
        )
    ).all()

    return [emp.id for emp in bd_employees]


def user_has_complete_company_details(user_id: str, db: Session) -> bool:
    """Check if user has completed company details."""
    company_details = db.query(CompanyDB).filter(CompanyDB.user_id == user_id).first()
    return company_details is not None and company_details.is_complete


def require_company_details(func):
    """Decorator to enforce company details completion on protected routes."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Extract request and db from the arguments
        request = None
        db = None

        # Find request in args or kwargs
        for arg in args:
            if hasattr(arg, 'url') and hasattr(arg, 'method'):  # Request object
                request = arg
                break
        if not request and 'request' in kwargs:
            request = kwargs['request']

        # Find db in kwargs
        if 'db' in kwargs:
            db = kwargs['db']

        if request and db:
            current_user = get_current_user(request, db)

            # Skip company details check for test sessions
            if current_user and hasattr(current_user, '_is_test_session') and current_user._is_test_session:
                return await func(*args, **kwargs)

            if current_user and not user_has_complete_company_details(current_user.id, db):
                return RedirectResponse(url="/company-details", status_code=302)

        return await func(*args, **kwargs)
    return wrapper


def require_pin_verification(func):
    """
    Decorator to enforce PIN verification on feature-locked routes.

    If a user has feature lock enabled and the current route is locked:
    - Check if PIN has been verified in the current session
    - If not verified, redirect to home_login with locked feature indicator
    - If verified, allow access
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Extract request and db from the arguments
        request = None
        db = None

        # Find request in args or kwargs
        for arg in args:
            if hasattr(arg, 'url') and hasattr(arg, 'method'):  # Request object
                request = arg
                break
        if not request and 'request' in kwargs:
            request = kwargs['request']

        # Find db in kwargs
        if 'db' in kwargs:
            db = kwargs['db']

        if request and db:
            current_user = get_current_user(request, db)

            # Check if user has feature lock enabled
            if current_user and current_user.feature_lock_enabled:
                # Get current route path
                current_path = request.url.path

                # Check if this route is locked
                locked_features = current_user.locked_features or []
                if current_path in locked_features:
                    # Check if PIN has been verified in session
                    session_token = request.cookies.get('session_token')
                    if session_token:
                        session_data = get_session(session_token)
                        pin_verified = session_data.get('pin_verified', False) if session_data else False

                        if not pin_verified:
                            # Redirect to home_login with locked feature parameter
                            return RedirectResponse(
                                url=f"/home_login?locked=true&feature={current_path}",
                                status_code=302
                            )

        return await func(*args, **kwargs)
    return wrapper


def require_bd_employee(func):
    """Decorator to enforce BD employee access only."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Extract request and db from the arguments
        request = None
        db = None

        # Find request in args or kwargs
        for arg in args:
            if hasattr(arg, 'url') and hasattr(arg, 'method'):  # Request object
                request = arg
                break
        if not request and 'request' in kwargs:
            request = kwargs['request']

        # Find db in kwargs
        if 'db' in kwargs:
            db = kwargs['db']

        if request and db:
            bd_employee = get_current_bd_employee(request, db)
            if not bd_employee:
                # Not a BD employee, redirect to employee login
                return RedirectResponse(url="/employee/login", status_code=302)

        return await func(*args, **kwargs)
    return wrapper


def block_bd_employees(func):
    """Decorator to block BD employees from accessing regular employee routes."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Extract request and db from the arguments
        request = None
        db = None

        # Find request in args or kwargs
        for arg in args:
            if hasattr(arg, 'url') and hasattr(arg, 'method'):  # Request object
                request = arg
                break
        if not request and 'request' in kwargs:
            request = kwargs['request']

        # Find db in kwargs
        if 'db' in kwargs:
            db = kwargs['db']

        if request and db:
            employee = get_current_employee(request, db)
            if employee and employee.is_bd:
                # This is a BD employee trying to access regular employee route
                return RedirectResponse(url="/bd/home", status_code=302)

        return await func(*args, **kwargs)
    return wrapper


def enforce_test_quarantine(func):
    """
    Decorator to enforce test session quarantine.
    If user is in a quarantined test session, redirect back to test endpoint.
    This prevents test users from escaping to other parts of the application.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Extract request and db from the arguments
        request = None
        db = None

        # Find request in args or kwargs
        for arg in args:
            if hasattr(arg, 'url') and hasattr(arg, 'method'):  # Request object
                request = arg
                break
        if not request and 'request' in kwargs:
            request = kwargs['request']

        # Find db in kwargs
        if 'db' in kwargs:
            db = kwargs['db']

        if request and db:
            current_user = get_current_user(request, db)

            # Check if this is a quarantined test session
            if current_user and hasattr(current_user, '_quarantined') and current_user._quarantined:
                # Build redirect URL back to test endpoint
                test_base_url = getattr(current_user, '_test_base_url', '/public-projects/nkbpl.pratyaksh')
                test_token = getattr(current_user, '_test_token', '')

                if test_token:
                    redirect_url = f"{test_base_url}?test_token={test_token}"
                else:
                    redirect_url = test_base_url

                # Redirect back to quarantined area
                return RedirectResponse(url=redirect_url, status_code=302)

        return await func(*args, **kwargs)
    return wrapper


# ==================== Expert-Verse Dependencies ====================


def get_current_expert(request: Request, db: Session = Depends(get_db)) -> Optional[ExpertDB]:
    """Get current expert from session."""
    session_token = request.cookies.get('expert_session_token')
    if not session_token:
        return None

    session_data = get_expert_session(session_token)
    if not session_data:
        return None

    expert = db.query(ExpertDB).filter(ExpertDB.id == session_data['expert_id']).first()
    return expert


def expert_has_complete_profile(expert_id: str, db: Session) -> bool:
    """Check if expert has completed their profile setup."""
    expert = db.query(ExpertDB).filter(ExpertDB.id == expert_id).first()
    if not expert:
        return False

    # Check if expert has marked profile as complete
    if not expert.profile_completed:
        return False

    # Verify at least basic profile info exists
    profile = db.query(ExpertProfileDB).filter(ExpertProfileDB.expert_id == expert_id).first()
    if not profile:
        return False

    # Check mandatory fields (expertise_areas and services_offered)
    if not profile.expertise_areas or not profile.services_offered:
        return False

    return True


def require_expert_login(func):
    """Decorator to enforce expert login on protected routes."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Extract request from arguments
        request = None
        for arg in args:
            if hasattr(arg, 'url') and hasattr(arg, 'method'):  # Request object
                request = arg
                break
        if not request and 'request' in kwargs:
            request = kwargs['request']

        if request:
            session_token = request.cookies.get('expert_session_token')
            if not session_token:
                return RedirectResponse(url="/expert/login", status_code=302)

            session_data = get_expert_session(session_token)
            if not session_data:
                return RedirectResponse(url="/expert/login", status_code=302)

        return await func(*args, **kwargs)
    return wrapper


def require_expert_profile_complete(func):
    """Decorator to enforce expert profile completion on protected routes."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Extract request and db from the arguments
        request = None
        db = None

        # Find request in args or kwargs
        for arg in args:
            if hasattr(arg, 'url') and hasattr(arg, 'method'):  # Request object
                request = arg
                break
        if not request and 'request' in kwargs:
            request = kwargs['request']

        # Find db in kwargs
        if 'db' in kwargs:
            db = kwargs['db']

        if request and db:
            current_expert = get_current_expert(request, db)
            if current_expert and not expert_has_complete_profile(current_expert.id, db):
                return RedirectResponse(url="/expert/profile-setup", status_code=302)

        return await func(*args, **kwargs)
    return wrapper
