import os
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Date, Boolean, Text, ForeignKey, Index, inspect, text, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship, backref
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import make_url
import logging
import uuid

logger = logging.getLogger(__name__)

DEFAULT_POSTGRES_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://tenderhub_admin:YOUR_PASSWORD@localhost:5432/tenderhub_db")


def _normalize_database_url(url: str) -> str:
    """Normalize DATABASE_URL for SQLAlchemy compatibility."""
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    if url.startswith("postgresql://") and "+psycopg" not in url:
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


# Database configuration
DATABASE_URL = _normalize_database_url(os.getenv("DATABASE_URL", DEFAULT_POSTGRES_URL))
database_url = make_url(DATABASE_URL)

# Connection pooling configuration
engine_kwargs: Dict[str, Any] = {
    "pool_pre_ping": True,
}

if database_url.get_backend_name() == "sqlite":
    engine_kwargs.update(
        connect_args={"check_same_thread": False},
        pool_size=1,
        max_overflow=0,
        pool_timeout=30,
        pool_recycle=3600,
    )
else:
    # PostgreSQL connection args with SSL support for Render
    # Note: For psycopg3, SSL is typically handled via the connection string
    # Render's DATABASE_URL usually includes SSL parameters already
    # We don't need to add connect_args for SSL unless explicitly required
    engine_kwargs.update(
        pool_size=int(os.getenv("DB_POOL_SIZE", "40")),
        max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "20")),
        pool_timeout=int(os.getenv("DB_POOL_TIMEOUT", "30")),
        pool_recycle=int(os.getenv("DB_POOL_RECYCLE", "1800")),
        echo_pool=False,
    )

engine = create_engine(DATABASE_URL, **engine_kwargs)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class TenderDB(Base):
    """SQLAlchemy model for tender data matching the scraper output format."""

    __tablename__ = "tenders"

    id = Column(String, primary_key=True, index=True)

    # Basic Details
    organisation_chain = Column(String, index=True)
    tender_reference_number = Column(String, index=True)
    tender_id = Column(String, unique=True, index=True)
    tender_type = Column(String, index=True)
    tender_category = Column(String, index=True)
    general_technical_evaluation_allowed = Column(String)
    payment_mode = Column(String)
    withdrawal_allowed = Column(String)
    form_of_contract = Column(String)
    no_of_covers = Column(String)
    itemwise_technical_evaluation_allowed = Column(String)
    is_multi_currency_allowed_for_boq = Column(String)
    is_multi_currency_allowed_for_fee = Column(String)
    allow_two_stage_bidding = Column(String)

    # Payment Instruments (JSON)
    payment_instruments = Column(JSONB)

    # Covers Information (JSON)
    covers_information = Column(JSONB)

    # Tender Fee Details (JSON)
    tender_fee_details = Column(JSONB)

    # EMD Fee Details (JSON)
    emd_fee_details = Column(JSONB)

    # Work Item Details (JSON)
    work_item_details = Column(JSONB)

    # Critical Dates (JSON)
    critical_dates = Column(JSONB)

    # Tender Documents (JSON)
    tender_documents = Column(JSONB)

    # Tender Inviting Authority (JSON)
    tender_inviting_authority = Column(JSONB)

    # Additional fields from scraper
    additional_fields = Column(JSONB)

    # Metadata
    scraped_at = Column(DateTime, default=datetime.utcnow)
    source_url = Column(String)
    source = Column(String, index=True)
    scraper_version = Column(String)
    search_term_used = Column(String, index=True)

    # Computed fields for frontend compatibility
    title = Column(String, index=True)
    authority = Column(String, index=True)
    state = Column(String, index=True)
    category = Column(String, index=True)
    estimated_value = Column(Float, index=True)
    currency = Column(String, default="INR")
    deadline = Column(DateTime, index=True)
    published_at = Column(DateTime, index=True)
    summary = Column(Text)
    pdf_url = Column(String)
    tags = Column(JSONB)
    awarded = Column(Boolean, default=False)
    awarded_at = Column(DateTime)
    awarded_by = Column(String, ForeignKey("users.id"))
    worked_by_name = Column(String, nullable=True)  # Name of person who awarded (admin or BD employee)
    worked_by_type = Column(String, nullable=True)  # 'user' or 'bd_employee'
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    favorites = relationship("FavoriteDB", back_populates="tender", cascade="all, delete-orphan")
    assignments = relationship("TenderAssignmentDB", back_populates="tender", cascade="all, delete-orphan")
    documents = relationship("TenderDocumentDB", back_populates="tender", cascade="all, delete-orphan")
    analysis_report = relationship("TenderAnalysisReportDB", back_populates="tender", uselist=False, cascade="all, delete-orphan")
    ai_insights = relationship("TenderAIInsightsDB", back_populates="tender", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_tender_deadline', 'deadline'),
        Index('idx_tender_published', 'published_at'),
        Index('idx_tender_value', 'estimated_value'),
        Index('idx_tender_search', 'title', 'authority', 'category'),
        Index('idx_tender_location', 'state'),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format matching the scraper output."""

        return {
            'Organisation Chain': self.organisation_chain,
            'Tender Reference Number': self.tender_reference_number,
            'Tender ID': self.tender_id,
            'Tender Type': self.tender_type,
            'Tender Category': self.tender_category,
            'General Technical Evaluation Allowed': self.general_technical_evaluation_allowed,
            'Payment Mode': self.payment_mode,
            'Withdrawal Allowed': self.withdrawal_allowed,
            'Form Of Contract': self.form_of_contract,
            'No. of Covers': self.no_of_covers,
            'ItemWise Technical Evaluation Allowed': self.itemwise_technical_evaluation_allowed,
            'Is Multi Currency Allowed For BOQ': self.is_multi_currency_allowed_for_boq,
            'Is Multi Currency Allowed For Fee': self.is_multi_currency_allowed_for_fee,
            'Allow Two Stage Bidding': self.allow_two_stage_bidding,
            'Payment Instruments': self.payment_instruments or {},
            'Covers Information': self.covers_information or [],
            'Tender Fee Details': self.tender_fee_details or {},
            'EMD Fee Details': self.emd_fee_details or {},
            'Work Item Details': self.work_item_details or {},
            'Critical Dates': self.critical_dates or {},
            'Tender Documents': self.tender_documents or {},
            'Tender Inviting Authority': self.tender_inviting_authority or {},
            'additional_fields': self.additional_fields or {},
            'scraping_metadata': {
                'scraped_at': self.scraped_at.isoformat() if self.scraped_at else None, #type: ignore
                'source_url': self.source_url,
                'scraper_version': self.scraper_version,
                'search_term_used': self.search_term_used
            }
        }

    def to_frontend_format(self) -> Dict[str, Any]:
        """Convert to frontend-compatible format."""

        return {
            'id': self.id,
            'title': self.title or '',
            'authority': self.authority or '',
            'state': self.state or '',
            'category': self.category or '',
            'estimated_value': self.estimated_value,
            'currency': self.currency,
            'deadline': self.deadline.isoformat() if self.deadline else None, #type: ignore
            'published_at': self.published_at.isoformat() if self.published_at else None, #type: ignore
            'tags': self.tags or [],
            'pdf_url': self.pdf_url or '',
            'summary': self.summary or '',
            'metadata': {
                'tender_reference_number': self.tender_reference_number,
                'tender_id': self.tender_id,
                'organisation_chain': self.organisation_chain,
                'tender_type': self.tender_type,
                'work_item_details': self.work_item_details,
                'critical_dates': self.critical_dates,
                'tender_fee_details': self.tender_fee_details,
                'emd_fee_details': self.emd_fee_details,
                'tender_documents': self.tender_documents,
                'tender_inviting_authority': self.tender_inviting_authority
            },
            'updated_at': self.updated_at.isoformat() if self.updated_at else None #type: ignore
        }


class TenderDocumentDB(Base):
    """SQLAlchemy model for storing tender documents and screenshots as binary data."""

    __tablename__ = "tender_documents"

    id = Column(Integer, primary_key=True, index=True)
    tender_id = Column(String, ForeignKey("tenders.id", ondelete="CASCADE"), nullable=False, index=True)

    # Document metadata
    document_type = Column(String, nullable=False, index=True)  # screenshot, pdf, excel, zip, etc.
    filename = Column(String, nullable=False)
    mime_type = Column(String, nullable=False)  # image/png, application/pdf, etc.
    file_size = Column(Integer, nullable=False)  # Size in bytes

    # Binary document data (nullable if stored in S3)
    file_data = Column(LargeBinary, nullable=True)

    # S3 storage fields
    s3_key = Column(String, nullable=True, index=True)  # S3 object key (e.g., 'tenders/123/document.pdf')
    s3_url = Column(String, nullable=True)  # S3 URL (for reference)
    migrated_to_s3 = Column(Boolean, default=False, nullable=False)  # Flag indicating file is in S3

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    display_order = Column(Integer, default=0)  # For ordering multiple screenshots

    # Relationship
    tender = relationship("TenderDB", back_populates="documents")

    # Indexes for query optimization
    __table_args__ = (
        Index('idx_tender_doc_type', 'tender_id', 'document_type'),
        Index('idx_tender_doc_order', 'tender_id', 'display_order'),
    )

    def to_dict(self, include_data: bool = False) -> Dict[str, Any]:
        """Convert to dictionary format. Set include_data=True to include binary data."""
        result = {
            'id': self.id,
            'tender_id': self.tender_id,
            'document_type': self.document_type,
            'filename': self.filename,
            'mime_type': self.mime_type,
            'file_size': self.file_size,
            's3_key': self.s3_key,
            's3_url': self.s3_url,
            'migrated_to_s3': self.migrated_to_s3,
            'created_at': self.created_at.isoformat() if self.created_at else None, #type: ignore
            'display_order': self.display_order,
        }
        if include_data:
            result['file_data'] = self.file_data
        return result


class UserDB(Base):
    """
    SQLAlchemy model for user accounts.
    """
    __tablename__ = "users"

    id = Column(String, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    company = Column(String)
    role = Column(String)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)

    # Profile Information
    profile_image = Column(String)  # Path to uploaded image
    phone_number = Column(String)
    job_title = Column(String)
    department = Column(String)
    bio = Column(Text)
    notification_preferences = Column(JSONB, server_default=text("'{}'"))  # Email notifications, etc.

    # Feature Lock System
    feature_lock_pin = Column(String, nullable=True)  # Hashed 6-digit PIN
    feature_lock_enabled = Column(Boolean, default=False)  # System active?
    locked_features = Column(JSONB, default=list)  # Array of locked route paths

    # Relationships
    favorites = relationship("FavoriteDB", back_populates="user")
    custom_cards = relationship("CustomCardDB", back_populates="user")
    projects = relationship("ProjectDB", back_populates="user")
    company_codes = relationship("CompanyCodeDB", back_populates="user")
    certificates = relationship("CertificateDB", back_populates="user")
    company_details = relationship("CompanyDB", back_populates="user", uselist=False)
    notifications = relationship("NotificationDB", foreign_keys="[NotificationDB.user_id]")
    filter_presets = relationship("FilterPresetDB", foreign_keys="[FilterPresetDB.user_id]")


class CompanyDB(Base):
    """
    SQLAlchemy model for company details.
    Required for government tender consultancy companies.
    """
    __tablename__ = "companies"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, unique=True)

    # Basic Company Information
    company_name = Column(String, nullable=False)
    registration_number = Column(String, nullable=False)  # CIN/Registration Number
    gst_number = Column(String, nullable=False)
    pan_number = Column(String, nullable=False)

    # Business Details
    industry_sector = Column(String, nullable=False)
    year_established = Column(Integer, nullable=False)
    annual_turnover = Column(String, nullable=False)  # e.g., "₹50-100 Crores"
    employee_count = Column(String, nullable=False)   # e.g., "250-500"

    # Contact Information
    registered_address = Column(Text, nullable=False)
    operational_address = Column(Text)
    phone_number = Column(String, nullable=False)
    email_address = Column(String, nullable=False)
    website_url = Column(String)

    # Government Tender Specific
    key_services = Column(Text, nullable=False)  # Services offered
    specialization_areas = Column(Text, nullable=False)  # Areas of expertise
    previous_govt_experience = Column(Text)  # Previous government projects
    certifications = Column(Text)  # Legacy storage for certifications (deprecated)

    # Structured details
    legal_details = Column(JSONB, default=dict)  # Licenses, registrations, compliance info
    financial_details = Column(JSONB, default=dict)  # Turnover history, financial metrics

    # Banking Details
    bank_name = Column(String, nullable=False, default='')
    account_number = Column(String, nullable=False, default='')
    ifsc_code = Column(String, nullable=False, default='')
    account_holder_name = Column(String, nullable=False, default='')

    # Key Personnel
    managing_director = Column(String, nullable=False)
    technical_head = Column(String)
    compliance_officer = Column(String)

    # Status and Metadata
    is_complete = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship
    user = relationship("UserDB", back_populates="company_details")
    company_certifications = relationship(
        "CompanyCertificateDB",
        back_populates="company",
        cascade="all, delete-orphan"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert company details to dictionary format."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'company_name': self.company_name,
            'registration_number': self.registration_number,
            'gst_number': self.gst_number,
            'pan_number': self.pan_number,
            'industry_sector': self.industry_sector,
            'year_established': self.year_established,
            'annual_turnover': self.annual_turnover,
            'employee_count': self.employee_count,
            'registered_address': self.registered_address,
            'operational_address': self.operational_address,
            'phone_number': self.phone_number,
            'email_address': self.email_address,
            'website_url': self.website_url,
            'key_services': self.key_services,
            'specialization_areas': self.specialization_areas,
            'previous_govt_experience': self.previous_govt_experience,
            'certifications': self.certifications,
            'legal_details': self.legal_details or {},
            'financial_details': self.financial_details or {},
            'bank_name': self.bank_name,
            'account_number': self.account_number,
            'ifsc_code': self.ifsc_code,
            'account_holder_name': self.account_holder_name,
            'managing_director': self.managing_director,
            'technical_head': self.technical_head,
            'compliance_officer': self.compliance_officer,
            'is_complete': self.is_complete,
            'created_at': self.created_at.isoformat() if self.created_at else None, #type: ignore
            'updated_at': self.updated_at.isoformat() if self.updated_at else None, #type: ignore
            'certification_records': [cert.to_dict() for cert in self.company_certifications]
        }


class CompanyCertificateDB(Base):
    """
    Stores supporting certificates uploaded as part of company profile.
    """
    __tablename__ = "company_certifications"

    id = Column(Integer, primary_key=True, index=True)
    company_id = Column(Integer, ForeignKey("companies.id", ondelete="CASCADE"), nullable=False)
    name = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    company = relationship("CompanyDB", back_populates="company_certifications")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "company_id": self.company_id,
            "name": self.name,
            "file_path": self.file_path,
            "uploaded_at": self.uploaded_at.isoformat() if self.uploaded_at else None, # type: ignore
        }

    __table_args__ = (
        Index('idx_company_certificates', 'company_id', 'uploaded_at'),
    )


class FavoriteDB(Base):
    """
    SQLAlchemy model for user favorites.
    """
    __tablename__ = "favorites"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    tender_id = Column(String, ForeignKey("tenders.id"), nullable=False)
    notes = Column(Text)
    status = Column(String, default="draft")  # draft, submitted, completed
    user_filled_data = Column(JSONB, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    worked_by_name = Column(String, nullable=True)  # Name of person who favorited (admin or BD employee)
    worked_by_type = Column(String, nullable=True)  # 'user' or 'bd_employee'

    # Relationships
    user = relationship("UserDB", back_populates="favorites")
    tender = relationship("TenderDB", back_populates="favorites")

    # Indexes for query optimization
    __table_args__ = (
        Index('idx_user_tender', 'user_id', 'tender_id', unique=True),
        Index('idx_user_favorites', 'user_id', 'created_at'),  # For user dashboard queries
    )


class CustomCardDB(Base):
    """
    SQLAlchemy model for custom search cards.
    """
    __tablename__ = "custom_cards"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    card_name = Column(String, nullable=False)
    core_search_terms = Column(String)
    state = Column(String)
    source = Column(String)  # Legacy field - keeping for backward compatibility
    sources = Column(JSONB)  # New field for multiple sources as JSON array
    tender_type = Column(String)
    sector = Column(String)  # New field for sector
    sub_sector = Column(String)  # New field for sub-sector
    work_type = Column(String)  # New field for work type (Works, Service, Goods, All)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("UserDB", back_populates="custom_cards")

    # Indexes for query optimization
    __table_args__ = (
        Index('idx_user_card_name', 'user_id', 'card_name', unique=True),
        Index('idx_user_cards', 'user_id'),  # For user card operations
    )


class ProjectDB(Base):
    """SQLAlchemy model for user projects."""
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)

    # Section 1: Project Details
    project_name = Column(String, nullable=False)
    project_description = Column(Text)
    complete_scope_of_work = Column(Text)  # Detailed scope of work for the project
    client_name = Column(String)
    sector = Column(String)  # Dropdown: Water Resources, Water Supply, Urban Infrastructure, Rural Infrastructure, Transportation, Energy, Forest
    sub_sector = Column(String)  # Dependent dropdown
    consultancy_fee = Column(Float)  # Indian Currency
    project_cost = Column(Float)  # Indian Currency
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    project_duration_months = Column(Integer)  # Auto calculated
    financing_authority = Column(Text, default="Financing Not Required")
    jv_partner = Column(String)
    country = Column(String)
    states = Column(JSONB)  # List of states (multiple)
    cities = Column(JSONB)  # List of cities (multiple)

    # Section 2: Services Rendered (JSON object with services as keys, each containing list of pointers)
    services_rendered = Column(JSONB, default=dict)

    # Section 3: Documents (JSON object with document types as keys, each containing list of file paths)
    documents = Column(JSONB, default=dict)

    # Auto-generation from awarded tenders
    source_tender_id = Column(String, nullable=True, index=True)  # Link to original TenderDB.id
    is_auto_generated = Column(Boolean, default=False)  # Flag for auto-created projects
    completion_status = Column(String, default="complete")  # 'incomplete' or 'complete' or 'completed_by_user'
    
    # Project ID: Format {COMPANY_INITIALS}-{NUMBER} (e.g., PCPL-000001)
    project_id = Column(String, unique=True, nullable=True, index=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("UserDB", back_populates="projects")

    # Indexes for query optimization
    __table_args__ = (
        Index('idx_user_project_name', 'user_id', 'project_name', unique=True),
        Index('idx_user_projects', 'user_id', 'created_at'),  # For project listings
        Index('idx_project_sector', 'sector'),  # For sector-based filtering
        Index('idx_project_dates', 'start_date', 'end_date'),  # For date-based queries
        Index('idx_source_tender', 'source_tender_id'),  # For tender-to-project lookups
        Index('idx_project_id', 'project_id'),  # For project ID lookups
    )


class CompanyCodeDB(Base):
    """
    SQLAlchemy model for company codes (for employee management).
    """
    __tablename__ = "company_codes"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    company_name = Column(String, nullable=False)
    company_code = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("UserDB", back_populates="company_codes")
    employees = relationship("EmployeeDB", back_populates="company_code")


class EmployeeDB(Base):
    """
    SQLAlchemy model for employees.
    """
    __tablename__ = "employees"

    id = Column(String, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    company_code_id = Column(Integer, ForeignKey("company_codes.id"), nullable=False)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    is_active = Column(Boolean, default=True)
    role = Column(String)
    team = Column(String)
    profile_picture = Column(String)
    is_bd = Column(Boolean, default=False)

    # Relationships
    company_code = relationship("CompanyCodeDB", back_populates="employees")
    assignments = relationship("TenderAssignmentDB", back_populates="employee")
    tasks = relationship("TaskDB", back_populates="employee")
    comments = relationship("TaskCommentDB", back_populates="employee")
    messages = relationship("TenderMessageDB", back_populates="employee")
    progress_updates = relationship("TaskProgressUpdateDB", back_populates="employee")
    owned_reports = relationship("TenderAnalysisReportDB", foreign_keys="TenderAnalysisReportDB.employee_id", back_populates="owner")
    edited_reports = relationship("TenderAnalysisReportDB", foreign_keys="TenderAnalysisReportDB.last_edited_by", back_populates="last_editor")
    report_attachments = relationship("ReportAttachmentDB", back_populates="uploader")


class TenderAssignmentDB(Base):
    """
    SQLAlchemy model for tender assignments to employees.
    """
    __tablename__ = "tender_assignments"

    id = Column(Integer, primary_key=True, index=True)
    tender_id = Column(String, ForeignKey("tenders.id", ondelete="CASCADE"), nullable=False)
    employee_id = Column(String, ForeignKey("employees.id"), nullable=False)
    role = Column(String, nullable=False)
    assigned_by = Column(String, ForeignKey("users.id"), nullable=False)
    priority = Column(String, default="medium")
    assigned_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    tender = relationship("TenderDB", back_populates="assignments")
    employee = relationship("EmployeeDB", back_populates="assignments")
    tasks = relationship("TaskDB", back_populates="assignment")
    messages = relationship("TenderMessageDB", back_populates="assignment")

    # Indexes for query optimization
    __table_args__ = (
        Index('idx_employee_assignments', 'employee_id', 'assigned_at'),  # For employee dashboard
        Index('idx_tender_assignments', 'tender_id'),  # For tender assignment queries
    )


class TaskDB(Base):
    """
    SQLAlchemy model for tasks within tender assignments.
    Supports 1-level subtask hierarchy (tasks can have subtasks, but subtasks cannot have children).
    """
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, index=True)
    assignment_id = Column(Integer, ForeignKey("tender_assignments.id"), nullable=False)
    title = Column(String, nullable=False)
    description = Column(Text)
    priority = Column(String, default="medium")
    status = Column(String, default="pending")
    estimated_hours = Column(Float)
    deadline = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    completed_by = Column(String, ForeignKey("employees.id"))

    # Subtask support (1-level deep only)
    parent_task_id = Column(Integer, ForeignKey("tasks.id"), nullable=True)
    is_subtask = Column(Boolean, default=False)
    subtask_order = Column(Integer, default=0)

    # Relationships
    assignment = relationship("TenderAssignmentDB", back_populates="tasks")
    employee = relationship("EmployeeDB", back_populates="tasks")
    comments = relationship("TaskCommentDB", back_populates="task")
    concerns = relationship("TaskConcernDB", back_populates="task", cascade="all, delete-orphan")
    files = relationship("TaskFileDB", back_populates="task", cascade="all, delete-orphan")
    progress_updates = relationship("TaskProgressUpdateDB", back_populates="task", cascade="all, delete-orphan", order_by="TaskProgressUpdateDB.created_at.desc()")

    # Self-referential relationship for subtasks
    subtasks = relationship("TaskDB", backref=backref("parent_task", remote_side=[id]), cascade="all, delete-orphan")

    # Indexes for query optimization
    __table_args__ = (
        Index('idx_assignment_tasks', 'assignment_id'),  # For task listings by assignment
        Index('idx_task_status', 'status', 'deadline'),  # For status-based queries
        Index('idx_parent_task', 'parent_task_id'),  # For subtask queries
    )


class StageTaskTemplateDB(Base):
    """
    SQLAlchemy model for task templates per tender stage.
    Used to automatically create tasks when stages progress.
    """
    __tablename__ = "stage_task_templates"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    stage_number = Column(Integer, nullable=False, index=True)  # 1-6

    # Task details
    task_title = Column(String, nullable=False)
    task_description = Column(Text)
    priority = Column(String, default="medium")  # low, medium, high
    estimated_hours = Column(Float)
    deadline_days = Column(Integer, nullable=False)  # Days after stage trigger

    # Subtask support
    parent_template_id = Column(Integer, ForeignKey("stage_task_templates.id"), nullable=True)
    is_subtask = Column(Boolean, default=False)
    task_order = Column(Integer, default=0)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("UserDB", foreign_keys=[user_id])
    subtasks = relationship("StageTaskTemplateDB",
                          backref=backref("parent_template", remote_side=[id]),
                          cascade="all, delete-orphan")

    # Indexes
    __table_args__ = (
        Index('idx_user_stage_templates', 'user_id', 'stage_number', 'task_order'),
        Index('idx_parent_template', 'parent_template_id'),
    )


class TaskCommentDB(Base):
    """
    SQLAlchemy model for task comments.
    """
    __tablename__ = "task_comments"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("tasks.id"), nullable=False)
    employee_id = Column(String, ForeignKey("employees.id"), nullable=False)
    comment = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    task = relationship("TaskDB", back_populates="comments")
    employee = relationship("EmployeeDB", back_populates="comments")


class TenderMessageDB(Base):
    """
    SQLAlchemy model for tender chat messages.
    """
    __tablename__ = "tender_messages"

    id = Column(Integer, primary_key=True, index=True)
    assignment_id = Column(Integer, ForeignKey("tender_assignments.id"), nullable=False)
    employee_id = Column(String, ForeignKey("employees.id"))  # NULL for manager messages
    message = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    assignment = relationship("TenderAssignmentDB", back_populates="messages")
    employee = relationship("EmployeeDB", back_populates="messages")


class TaskConcernDB(Base):
    """
    SQLAlchemy model for task-specific concerns/requests raised by employees.
    Allows employees to flag issues with tasks that need manager attention.
    """
    __tablename__ = "task_concerns"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False)
    employee_id = Column(String, ForeignKey("employees.id"), nullable=False)
    concern_type = Column(String, nullable=False)  # timeline, resources, clarification, blocker, other
    title = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    status = Column(String, default="open")  # open, acknowledged, resolved, closed
    priority = Column(String, default="medium")  # low, medium, high, urgent
    resolved_by = Column(String, ForeignKey("users.id"), nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    resolution_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    task = relationship("TaskDB", back_populates="concerns")
    employee = relationship("EmployeeDB", foreign_keys=[employee_id])
    resolver = relationship("UserDB", foreign_keys=[resolved_by])

    # Indexes for query optimization
    __table_args__ = (
        Index('idx_task_concerns', 'task_id', 'status'),
        Index('idx_employee_concerns', 'employee_id', 'created_at'),
    )


class TaskFileDB(Base):
    """
    SQLAlchemy model for task deliverable files uploaded by employees.
    Stores binary file data for deliverables, reports, and other task-related documents.
    """
    __tablename__ = "task_files"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False)
    employee_id = Column(String, ForeignKey("employees.id"), nullable=False)
    
    # File metadata
    filename = Column(String, nullable=False)
    mime_type = Column(String, nullable=False)  # application/pdf, image/png, etc.
    file_size = Column(Integer, nullable=False)  # Size in bytes
    file_data = Column(LargeBinary, nullable=False)  # Binary file data
    
    # File description/notes
    description = Column(Text, nullable=True)  # Optional description from employee
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    task = relationship("TaskDB", back_populates="files")
    employee = relationship("EmployeeDB")

    # Indexes for query optimization
    __table_args__ = (
        Index('idx_task_files', 'task_id', 'created_at'),
        Index('idx_employee_files', 'employee_id', 'created_at'),
    )


class TaskProgressUpdateDB(Base):
    """
    SQLAlchemy model for task progress updates/work logs.
    Allows employees to track step-by-step progress within a task.
    """
    __tablename__ = "task_progress_updates"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False)
    employee_id = Column(String, ForeignKey("employees.id"), nullable=False)

    # Progress update content
    update_text = Column(Text, nullable=False)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    edited_at = Column(DateTime, nullable=True)
    is_edited = Column(Boolean, default=False)

    # Relationships
    task = relationship("TaskDB", back_populates="progress_updates")
    employee = relationship("EmployeeDB", back_populates="progress_updates")

    # Indexes for query optimization
    __table_args__ = (
        Index('idx_task_progress', 'task_id', 'created_at'),
        Index('idx_employee_progress', 'employee_id', 'created_at'),
    )


class EmployeeNotificationDB(Base):
    """
    SQLAlchemy model for employee-specific notifications.
    Separate from UserDB notifications to allow different notification types and behaviors.
    """
    __tablename__ = "employee_notifications"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(String, ForeignKey("employees.id", ondelete="CASCADE"), nullable=False)
    notification_type = Column(String, nullable=False)  # task_assigned, task_updated, task_deleted, concern_resolved, etc.
    title = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    related_task_id = Column(Integer, ForeignKey("tasks.id", ondelete="SET NULL"), nullable=True)
    related_tender_id = Column(String, ForeignKey("tenders.id", ondelete="SET NULL"), nullable=True)
    is_read = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    employee = relationship("EmployeeDB", foreign_keys=[employee_id])
    task = relationship("TaskDB", foreign_keys=[related_task_id])
    tender = relationship("TenderDB", foreign_keys=[related_tender_id])

    # Indexes for query optimization
    __table_args__ = (
        Index('idx_employee_notifications', 'employee_id', 'is_read', 'created_at'),
        Index('idx_notification_type', 'notification_type'),
    )


class CertificateDB(Base):
    """
    SQLAlchemy model for processed certificates.
    """
    __tablename__ = "certificates"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    batch_id = Column(String, ForeignKey("bulk_upload_batches.id"), nullable=True)  # Link to batch

    # Extracted certificate data
    project_name = Column(String, nullable=False)
    client_name = Column(String)
    completion_date = Column(DateTime)
    project_value = Column(Float)
    project_value_inr = Column(String)
    services_rendered = Column(JSONB)  # List of services
    location = Column(String)
    sectors = Column(JSONB)  # List of sectors
    sub_sectors = Column(JSONB)  # List of subsectors
    consultancy_fee_inr = Column(String)
    consultancy_fee_numeric = Column(Float, index=True)  # Parsed numeric fee for efficient filtering
    scope_of_work = Column(Text)
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    duration = Column(String)
    issuing_authority_details = Column(Text)
    performance_remarks = Column(Text)
    certificate_number = Column(String)
    signing_authority_details = Column(Text)
    role_lead_jv = Column(String)
    jv_partners = Column(JSONB)
    funding_agency = Column(String)
    confidence_score = Column(Float)
    metrics = Column(JSONB)  # List of metric objects
    verbatim_certificate = Column(Text)

    # Additional metadata
    original_filename = Column(String)
    file_path = Column(String)  # Local file path (legacy, may not exist on Render)
    s3_key = Column(String, nullable=True, index=True)  # S3 object key (e.g., 'certificates/user_id/batch_id/filename')
    s3_url = Column(String, nullable=True)  # S3 URL for direct access
    file_hash = Column(String, index=True)  # SHA256 hash for duplicate detection
    file_size = Column(Integer)  # File size in bytes
    extracted_text = Column(Text)
    processing_status = Column(String, default="processing")  # processing, completed, failed, duplicate, extraction_failed
    processing_error = Column(Text)
    
    # Extraction metadata for multi-tier processing
    extraction_method = Column(String)  # "primary_vision", "enhanced_vision", "premium_vision", "all_methods_failed"
    parsing_method = Column(String)  # "gpt-4o", "gpt-4o-enhanced", "o1-mini", "parsing_failed"
    extraction_quality_score = Column(Float)  # 0-1 confidence score for extraction quality
    extraction_attempts = Column(Integer, default=1)  # Number of processing attempts

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime)

    # Relationships
    user = relationship("UserDB", back_populates="certificates")
    vectors = relationship("VectorDB", back_populates="certificate")
    batch = relationship("BulkUploadBatchDB", back_populates="certificates")

    # Indexes for query optimization
    __table_args__ = (
        Index('idx_user_certificates', 'user_id', 'created_at'),  # For certificate searches
        Index('idx_certificate_status', 'processing_status'),  # For status filtering
        Index('idx_batch_certificates', 'batch_id'),  # For batch queries
        Index('idx_user_file_hash', 'user_id', 'file_hash'),  # For duplicate detection
    )


class TenderCertificateAttachmentDB(Base):
    """
    SQLAlchemy model for certificate-tender attachments.
    Links completion certificates to tenders throughout the tender lifecycle.
    """
    __tablename__ = "tender_certificate_attachments"

    id = Column(Integer, primary_key=True, index=True)

    # Foreign Keys
    certificate_id = Column(String, ForeignKey("certificates.id", ondelete="CASCADE"), nullable=False)
    tender_id = Column(String, ForeignKey("tenders.id", ondelete="CASCADE"), nullable=False)
    attached_by_user_id = Column(String, ForeignKey("users.id"), nullable=False)

    # Metadata
    attached_by_type = Column(String, nullable=False)  # 'user' or 'bd_employee'
    attached_by_name = Column(String, nullable=True)   # Display name
    attached_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    notes = Column(Text, nullable=True)  # Optional notes about why attached

    # Relationships
    certificate = relationship("CertificateDB", backref="tender_attachments")
    tender = relationship("TenderDB", backref="certificate_attachments")
    user = relationship("UserDB", foreign_keys=[attached_by_user_id])

    # Indexes
    __table_args__ = (
        # Unique constraint: one certificate can only be attached once to a tender
        Index('idx_unique_cert_tender', 'certificate_id', 'tender_id', unique=True),
        # Query index for finding attachments by tender
        Index('idx_tender_attachments', 'tender_id', 'attached_at'),
        # Query index for finding attachments by certificate
        Index('idx_certificate_attachments', 'certificate_id'),
    )

    def __repr__(self):
        return f"<TenderCertificateAttachment(id={self.id}, cert={self.certificate_id[:8]}, tender={self.tender_id[:8]})>"


class VectorDB(Base):
    """
    SQLAlchemy model for certificate embeddings and vector search.
    """
    __tablename__ = "certificate_vectors"

    id = Column(Integer, primary_key=True, index=True)
    certificate_id = Column(String, ForeignKey("certificates.id"), nullable=False)

    # Embedding data
    embedding = Column(JSONB)  # Store as JSON array for FAISS compatibility
    embedding_model = Column(String, default="text-embedding-ada-002")
    content_type = Column(String, default="full_text")  # full_text, project_name, etc.

    # FAISS index data
    faiss_index_path = Column(String)  # Path to FAISS index file

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    certificate = relationship("CertificateDB", back_populates="vectors")


class BulkUploadBatchDB(Base):
    """
    SQLAlchemy model for tracking bulk certificate upload batches.
    """
    __tablename__ = "bulk_upload_batches"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)

    # Batch metadata
    batch_name = Column(String)  # Optional name for the batch
    upload_type = Column(String, nullable=False)  # 'files', 'folder', 'zip'

    # Progress tracking
    total_files = Column(Integer, default=0)
    processed_count = Column(Integer, default=0)
    success_count = Column(Integer, default=0)
    failed_count = Column(Integer, default=0)
    skipped_count = Column(Integer, default=0)  # Duplicates skipped
    in_progress_count = Column(Integer, default=0)

    # Status: 'queued', 'processing', 'completed', 'failed'
    status = Column(String, default='queued', index=True)

    # Error tracking
    errors = Column(JSONB, default=list)  # List of error messages

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)

    # Relationships
    user = relationship("UserDB", foreign_keys=[user_id])
    certificates = relationship("CertificateDB", back_populates="batch")

    # Indexes
    __table_args__ = (
        Index('idx_user_batches', 'user_id', 'created_at'),
        Index('idx_batch_status', 'status', 'created_at'),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'batch_name': self.batch_name,
            'upload_type': self.upload_type,
            'total_files': self.total_files,
            'processed_count': self.processed_count,
            'success_count': self.success_count,
            'failed_count': self.failed_count,
            'skipped_count': self.skipped_count,
            'in_progress_count': self.in_progress_count,
            'status': self.status,
            'errors': self.errors or [],
            'created_at': self.created_at.isoformat() if self.created_at else None,  # type: ignore
            'started_at': self.started_at.isoformat() if self.started_at else None,  # type: ignore
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,  # type: ignore
        }


class FilterPresetDB(Base):
    """
    SQLAlchemy model for storing certificate filter presets.
    Allows users to save commonly used filter combinations for quick access.
    """
    __tablename__ = "certificate_filter_presets"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)

    # Preset metadata
    preset_name = Column(String, nullable=False)
    filters = Column(JSONB, nullable=False)  # Complete filter object with all selected values
    is_default = Column(Boolean, default=False)  # Mark one preset as default

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("UserDB", foreign_keys=[user_id])

    # Indexes
    __table_args__ = (
        Index('idx_user_presets', 'user_id', 'created_at'),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'preset_name': self.preset_name,
            'filters': self.filters,
            'is_default': self.is_default,
            'created_at': self.created_at.isoformat() if self.created_at else None,  # type: ignore
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,  # type: ignore
        }


class ShortlistedTenderDB(Base):
    """
    SQLAlchemy model for shortlisted tenders.
    """
    __tablename__ = "shortlisted_tenders"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    tender_id = Column(String, ForeignKey("tenders.id", ondelete="CASCADE"), nullable=False)
    reason = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Progress tracking fields
    # progress_data stores: step1-6 status values, step_timestamps dict,
    # step4_deadline and step5_deadline (YYYY-MM-DD format) for manual deadlines
    progress_data = Column(JSONB, default=dict)
    worked_by_name = Column(String, nullable=True)  # Name of person who shortlisted (admin or BD employee)
    worked_by_type = Column(String, nullable=True)  # 'user' or 'bd_employee'

    # Relationships
    tender = relationship("TenderDB", foreign_keys=[tender_id])
    user = relationship("UserDB", foreign_keys=[user_id])

    # Unique constraint
    __table_args__ = (
        Index('idx_user_tender_shortlist', 'user_id', 'tender_id', unique=True),
    )


class RejectedTenderDB(Base):
    """
    SQLAlchemy model for rejected tenders.
    """
    __tablename__ = "rejected_tenders"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    tender_id = Column(String, ForeignKey("tenders.id", ondelete="CASCADE"), nullable=False)
    reason = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    tender = relationship("TenderDB", foreign_keys=[tender_id])
    user = relationship("UserDB", foreign_keys=[user_id])

    # Unique constraint
    __table_args__ = (
        Index('idx_user_tender_reject', 'user_id', 'tender_id', unique=True),
    )


class NotificationDB(Base):
    """
    SQLAlchemy model for user notifications.
    Tracks deadline notifications for favorited and shortlisted tenders.
    """
    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    tender_id = Column(String, ForeignKey("tenders.id", ondelete="CASCADE"), nullable=False)
    notification_type = Column(String, nullable=False)  # deadline_10d, deadline_7d, deadline_5d, deadline_2d, deadline_today, reminder
    message = Column(Text, nullable=False)
    tender_title = Column(String)  # Denormalized for faster display
    days_remaining = Column(Integer)  # How many days until deadline
    is_read = Column(Boolean, default=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    tender = relationship("TenderDB", foreign_keys=[tender_id])
    user = relationship("UserDB", foreign_keys=[user_id], overlaps="notifications")

    # Indexes for query optimization
    __table_args__ = (
        Index('idx_user_notifications', 'user_id', 'is_read', 'created_at'),
        Index('idx_tender_notification_type', 'tender_id', 'notification_type', unique=True),
    )


class ReminderDB(Base):
    """
    SQLAlchemy model for user-set reminders on tenders.
    """
    __tablename__ = "reminders"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    tender_id = Column(String, ForeignKey("tenders.id", ondelete="CASCADE"), nullable=False)
    reminder_datetime = Column(DateTime, nullable=False, index=True)
    title = Column(String, nullable=False)  # Denormalized tender title
    note = Column(Text)  # Optional note for the reminder
    is_triggered = Column(Boolean, default=False, index=True)
    is_dismissed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    tender = relationship("TenderDB", foreign_keys=[tender_id])
    user = relationship("UserDB", foreign_keys=[user_id])

    # Indexes for query optimization
    __table_args__ = (
        Index('idx_user_reminders', 'user_id', 'reminder_datetime'),
        Index('idx_pending_reminders', 'is_triggered', 'reminder_datetime'),
    )


class CalendarActivityDB(Base):
    """
    SQLAlchemy model for persistent calendar activities.
    Activities are stored independently of tenders/projects/reminders
    so they remain on the calendar even after the source is deleted.
    """
    __tablename__ = "calendar_activities"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)

    # Activity metadata
    activity_date = Column(DateTime, nullable=False, index=True)
    activity_type = Column(String, nullable=False, index=True)  # deadline, activity, reminder
    title = Column(String, nullable=False)
    description = Column(Text)

    # Source tracking (optional, can be null if source deleted)
    tender_id = Column(String, ForeignKey("tenders.id", ondelete="SET NULL"), nullable=True)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="SET NULL"), nullable=True)
    reminder_id = Column(Integer, ForeignKey("reminders.id", ondelete="SET NULL"), nullable=True)

    # Status tracking
    source_deleted = Column(Boolean, default=False, index=True)  # True if original tender/project/reminder deleted
    is_active = Column(Boolean, default=True, index=True)  # Can be manually archived

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    deleted_at = Column(DateTime)  # When the source was deleted

    # Relationships
    user = relationship("UserDB", foreign_keys=[user_id])
    tender = relationship("TenderDB", foreign_keys=[tender_id])
    project = relationship("ProjectDB", foreign_keys=[project_id])
    reminder = relationship("ReminderDB", foreign_keys=[reminder_id])

    # Indexes for query optimization
    __table_args__ = (
        Index('idx_user_activity_date', 'user_id', 'activity_date'),
        Index('idx_activity_type_date', 'activity_type', 'activity_date'),
        Index('idx_user_active_activities', 'user_id', 'is_active', 'activity_date'),
    )


class SeenTenderDB(Base):
    """
    SQLAlchemy model for tracking which tenders users and BD employees have viewed.
    """
    __tablename__ = "seen_tenders"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)
    employee_id = Column(String, ForeignKey("employees.id", ondelete="CASCADE"), nullable=True, index=True)
    tender_id = Column(String, ForeignKey("tenders.id", ondelete="CASCADE"), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    user = relationship("UserDB", foreign_keys=[user_id])
    employee = relationship("EmployeeDB", foreign_keys=[employee_id])
    tender = relationship("TenderDB", foreign_keys=[tender_id])

    # Unique constraints and indexes
    __table_args__ = (
        Index('idx_user_tender_seen', 'user_id', 'tender_id', unique=True),
        Index('idx_employee_tender_seen', 'employee_id', 'tender_id', unique=True),
    )


class DumpedTenderDB(Base):
    """
    SQLAlchemy model for dumped/killed tenders (incomplete applications).
    """
    __tablename__ = "dumped_tenders"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    tender_id = Column(String, ForeignKey("tenders.id", ondelete="CASCADE"), nullable=False)
    shortlist_reason = Column(Text)  # Original reason for shortlisting
    kill_reason = Column(Text, nullable=False)  # Reason for killing/dumping
    kill_stage = Column(String)  # At which stage was it killed
    progress_data = Column(JSONB)  # All progress data up to kill point
    created_at = Column(DateTime)  # Original shortlist date
    killed_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    tender = relationship("TenderDB", foreign_keys=[tender_id])
    user = relationship("UserDB", foreign_keys=[user_id])


class StageDocumentDB(Base):
    """
    SQLAlchemy model for documents uploaded at each progress stage.
    """
    __tablename__ = "stage_documents"

    id = Column(Integer, primary_key=True, index=True)
    shortlist_id = Column(Integer, ForeignKey("shortlisted_tenders.id"), nullable=False)
    step_number = Column(Integer, nullable=False)  # Step number 1-6
    title = Column(String, nullable=False)  # Document title
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_size = Column(Integer)  # File size in bytes
    notes = Column(Text)  # Optional notes
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    shortlisted_tender = relationship("ShortlistedTenderDB", foreign_keys=[shortlist_id])

    # Index for efficient queries
    __table_args__ = (
        Index('idx_shortlist_step', 'shortlist_id', 'step_number'),
    )


class TenderResponseDB(Base):
    """
    SQLAlchemy model for tender responses.
    Each response represents a complete submission package for a tender.
    """
    __tablename__ = "tender_responses"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    tender_id = Column(String, ForeignKey("tenders.id", ondelete="CASCADE"), nullable=False)

    # Response details
    response_name = Column(String, nullable=False)
    response_type = Column(String, nullable=False)  # e.g., "Technical Bid", "Financial Bid", "Complete Proposal"
    remarks = Column(Text)

    # Sign and stamp
    signature_path = Column(String)  # Path to signature image
    stamp_path = Column(String)  # Path to stamp image

    # Generated PDF
    pdf_path = Column(String)  # Path to generated PDF
    pdf_filename = Column(String)  # Original filename for download

    # Status
    is_finalized = Column(Boolean, default=False)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    finalized_at = Column(DateTime)

    # Relationships
    tender = relationship("TenderDB", foreign_keys=[tender_id])
    user = relationship("UserDB", foreign_keys=[user_id])
    documents = relationship("ResponseDocumentDB", back_populates="response", cascade="all, delete-orphan")

    # Indexes for query optimization
    __table_args__ = (
        Index('idx_user_tender_responses', 'user_id', 'tender_id', 'created_at'),
        Index('idx_tender_responses', 'tender_id'),
    )


class ResponseDocumentDB(Base):
    """
    SQLAlchemy model for documents within a tender response.
    """
    __tablename__ = "response_documents"

    id = Column(Integer, primary_key=True, index=True)
    response_id = Column(Integer, ForeignKey("tender_responses.id"), nullable=False)

    # Document details
    document_name = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_size = Column(Integer)  # File size in bytes
    file_type = Column(String)  # MIME type

    # Order in the response
    display_order = Column(Integer, default=0)

    # Timestamps
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    response = relationship("TenderResponseDB", back_populates="documents")

    # Index for efficient queries
    __table_args__ = (
        Index('idx_response_documents', 'response_id', 'display_order'),
    )


class TenderAnalysisStatusDB(Base):
    """
    SQLAlchemy model for tracking tender eligibility analysis status.
    Used to cache analysis results and avoid re-processing the same tender.
    """
    __tablename__ = "tender_analysis_status"

    id = Column(String, primary_key=True, index=True)
    tender_id = Column(String, ForeignKey("tenders.id", ondelete="CASCADE"), nullable=False, unique=True, index=True)

    # Analysis status
    status = Column(String, nullable=False, default='not_started', index=True)  # not_started, processing, completed, failed

    # Progress tracking
    total_pages = Column(Integer, default=0)
    processed_pages = Column(Integer, default=0)
    total_criteria_found = Column(Integer, default=0)

    # Error tracking
    error_message = Column(Text)

    # Timestamps
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    tender = relationship("TenderDB", foreign_keys=[tender_id])

    # Indexes for efficient queries
    __table_args__ = (
        Index('idx_tender_analysis_status', 'tender_id', 'status'),
    )


class TenderEligibilityCriteriaDB(Base):
    """
    SQLAlchemy model for storing extracted technical eligibility criteria from tender documents.
    Each criterion represents a specific requirement that may need certificate proof.
    """
    __tablename__ = "tender_eligibility_criteria"

    id = Column(String, primary_key=True, index=True)
    tender_id = Column(String, ForeignKey("tenders.id", ondelete="CASCADE"), nullable=False, index=True)

    # Page information
    page_number = Column(Integer, nullable=False)  # Which page this criterion was extracted from

    # Criterion classification
    category = Column(String, nullable=False, index=True)  # authority, scope_of_work, metrics, financial, experience, location, services, technical, other
    criteria_type = Column(String, default='mandatory')  # mandatory, desirable, scoring

    # Criterion content
    criteria_text = Column(Text, nullable=False)  # Full verbatim text of the criterion
    extracted_requirements = Column(JSONB, default=dict)  # Structured requirements (e.g., {"min_value": 5, "unit": "years", "type": "experience"})
    keywords = Column(JSONB, default=list)  # Extracted keywords for fast matching

    # Embedding for semantic search
    embedding = Column(JSONB)  # 3072-dim vector from text-embedding-3-large

    # Processing metadata
    processing_status = Column(String, default='completed')  # pending, completed, failed
    confidence_score = Column(Float, default=0.0)  # 0.0 to 1.0

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    tender = relationship("TenderDB", foreign_keys=[tender_id])
    matches = relationship("TenderCertificateMatchDB", back_populates="criteria", cascade="all, delete-orphan")

    # Indexes for efficient queries
    __table_args__ = (
        Index('idx_tender_criteria_category', 'tender_id', 'category'),
        Index('idx_criteria_page', 'tender_id', 'page_number'),
    )


class TenderCertificateMatchDB(Base):
    """
    SQLAlchemy model for storing matches between tender criteria and user certificates.
    Represents that a specific certificate satisfies a specific tender criterion.
    """
    __tablename__ = "tender_certificate_matches"

    id = Column(Integer, primary_key=True, index=True)
    tender_id = Column(String, ForeignKey("tenders.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    certificate_id = Column(String, ForeignKey("certificates.id", ondelete="CASCADE"), nullable=False, index=True)
    criteria_id = Column(String, ForeignKey("tender_eligibility_criteria.id", ondelete="CASCADE"), nullable=False, index=True)

    # Match classification
    category = Column(String, nullable=False, index=True)  # Same as criterion category
    match_type = Column(String, nullable=False)  # full, partial, semantic, gpt_validated
    match_score = Column(Float, nullable=False, default=0.0)  # 0.0 to 1.0

    # Match details
    match_details = Column(JSONB, default=dict)  # Detailed breakdown of what matched (e.g., {"matched_fields": ["authority", "location"], "partial_match_count": "1/4"})
    validation_notes = Column(Text)  # GPT's explanation of why this is a match

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    tender = relationship("TenderDB", foreign_keys=[tender_id])
    user = relationship("UserDB", foreign_keys=[user_id])
    certificate = relationship("CertificateDB", foreign_keys=[certificate_id])
    criteria = relationship("TenderEligibilityCriteriaDB", back_populates="matches")

    # Indexes for efficient queries
    __table_args__ = (
        Index('idx_tender_user_matches', 'tender_id', 'user_id'),
        Index('idx_certificate_matches', 'certificate_id'),
        Index('idx_criteria_matches', 'criteria_id'),
        Index('idx_match_category', 'tender_id', 'category'),
        # Unique constraint: one match record per certificate-criteria pair
        Index('idx_unique_cert_criteria_match', 'certificate_id', 'criteria_id', unique=True),
    )


class TenderMatchDB(Base):
    """High-level tender-to-certificate match runs (per tender/user)."""

    __tablename__ = "tender_matches"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    tender_id = Column(String, ForeignKey("tenders.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    tender_name = Column(String, nullable=False)
    tender_number = Column(String)
    client_authority = Column(String)

    required_sectors = Column(JSONB, default=list)
    required_subsectors = Column(JSONB, default=list)
    required_services = Column(JSONB, default=list)
    min_project_value = Column(Float)
    min_consultancy_fee = Column(Float)
    similar_works_count = Column(Integer, default=0)
    years_lookback = Column(Integer, default=7)
    location_requirement = Column(JSONB, default=list)
    technical_metrics = Column(JSONB, default=list)
    funding_agencies = Column(JSONB, default=list)
    score_threshold = Column(Float, default=50.0)
    summary = Column(JSONB, default=dict)
    notes = Column(Text)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    tender = relationship("TenderDB", foreign_keys=[tender_id])
    user = relationship("UserDB", foreign_keys=[user_id])
    results = relationship("TenderMatchResultDB", back_populates="match", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_tender_matches_tender_user', 'tender_id', 'user_id'),
    )


class TenderMatchResultDB(Base):
    """Per-certificate scoring outcomes for a given tender match run."""

    __tablename__ = "tender_match_results"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    match_id = Column(String, ForeignKey("tender_matches.id", ondelete="CASCADE"), nullable=False, index=True)
    certificate_id = Column(String, ForeignKey("certificates.id", ondelete="CASCADE"), nullable=False, index=True)

    score = Column(Float, nullable=False)
    breakdown = Column(JSONB, default=dict)
    matching_factors = Column(JSONB, default=list)
    gaps = Column(JSONB, default=list)
    include_in_report = Column(Boolean, default=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    match = relationship("TenderMatchDB", back_populates="results")
    certificate = relationship("CertificateDB")

    __table_args__ = (
        Index('idx_match_certificate_unique', 'match_id', 'certificate_id', unique=True),
    )


# ==================== Expert-Verse Database Models ====================


class ExpertDB(Base):
    """Main expert user table for Expert-Verse feature."""

    __tablename__ = "experts"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    email = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    is_active = Column(Boolean, default=True)
    profile_completed = Column(Boolean, default=False, index=True)

    # Profile information
    profile_image = Column(String)
    phone_number = Column(String)
    bio = Column(Text)
    hourly_rate = Column(Float)
    availability_status = Column(String, default='available')  # available/busy/unavailable
    languages = Column(JSONB, default=list)

    # Computed/aggregate fields
    rating_average = Column(Float, default=0.0)
    total_projects = Column(Integer, default=0)
    total_earnings = Column(Float, default=0.0)

    # Relationships
    profile = relationship("ExpertProfileDB", back_populates="expert", uselist=False)
    content = relationship("ExpertContentDB", back_populates="expert")
    service_requests = relationship("ExpertServiceRequestDB", back_populates="expert")
    applications = relationship("ExpertApplicationDB", back_populates="expert")
    reviews = relationship("ExpertReviewDB", back_populates="expert")
    payments = relationship("ExpertPaymentDB", back_populates="expert")
    notifications = relationship("ExpertNotificationDB", back_populates="expert")
    favorite_tenders = relationship("ExpertFavoriteTenderDB", back_populates="expert")


class ExpertProfileDB(Base):
    """Extended expert profile information."""

    __tablename__ = "expert_profiles"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    expert_id = Column(String, ForeignKey("experts.id", ondelete="CASCADE"), nullable=False, unique=True, index=True)

    # Core professional info
    expertise_areas = Column(JSONB, default=list)  # ["Government Tenders", "Infrastructure", etc]
    services_offered = Column(JSONB, default=list)  # ["Bid Writing", "Compliance Review", etc]
    employment_type = Column(String)  # freelance/employed/both
    experience_years = Column(Integer)

    # Education & Qualifications
    qualifications = Column(JSONB, default=list)  # [{"degree": "MBA", "institution": "...", "year": 2020}]
    education = Column(JSONB, default=list)  # Detailed education history
    certifications = Column(JSONB, default=list)  # [{"name": "PMP", "issuer": "PMI", "file_path": "..."}]

    # Work history
    past_employers = Column(JSONB, default=list)

    # Online presence
    portfolio_url = Column(String)
    linkedin_url = Column(String)
    github_url = Column(String)
    website_url = Column(String)

    # Availability
    location = Column(String)
    willing_to_travel = Column(Boolean, default=False)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship
    expert = relationship("ExpertDB", back_populates="profile")


class ExpertContentDB(Base):
    """Expert Wall content - blogs, articles, case studies, whitepapers."""

    __tablename__ = "expert_content"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    expert_id = Column(String, ForeignKey("experts.id", ondelete="CASCADE"), nullable=False, index=True)

    # Content details
    title = Column(String, nullable=False)
    slug = Column(String, unique=True, nullable=False, index=True)  # SEO-friendly URL
    content = Column(Text, nullable=False)  # Markdown format
    content_html = Column(Text)  # Rendered HTML
    content_type = Column(String, default='blog', index=True)  # blog/case_study/whitepaper/qa
    excerpt = Column(String)  # Short summary
    featured_image = Column(String)

    # Status & visibility
    status = Column(String, default='draft', index=True)  # draft/published

    # Engagement metrics
    view_count = Column(Integer, default=0)
    like_count = Column(Integer, default=0)
    comment_count = Column(Integer, default=0)

    # Organization
    tags = Column(JSONB, default=list)

    # SEO
    seo_description = Column(String)

    # Timestamps
    published_at = Column(DateTime, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    expert = relationship("ExpertDB", back_populates="content")
    comments = relationship("ExpertContentCommentDB", back_populates="content", cascade="all, delete-orphan")
    likes = relationship("ExpertContentLikeDB", back_populates="content", cascade="all, delete-orphan")


class ExpertContentCommentDB(Base):
    """Comments on expert content."""

    __tablename__ = "expert_content_comments"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    content_id = Column(String, ForeignKey("expert_content.id", ondelete="CASCADE"), nullable=False, index=True)

    # Commenter (can be user, expert, or guest)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), index=True)
    expert_id = Column(String, ForeignKey("experts.id", ondelete="CASCADE"), index=True)
    commenter_name = Column(String)  # For guests

    # Comment data
    comment = Column(Text, nullable=False)
    is_edited = Column(Boolean, default=False)

    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    content = relationship("ExpertContentDB", back_populates="comments")
    user = relationship("UserDB")
    expert = relationship("ExpertDB")


class ExpertContentLikeDB(Base):
    """Likes on expert content."""

    __tablename__ = "expert_content_likes"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    content_id = Column(String, ForeignKey("expert_content.id", ondelete="CASCADE"), nullable=False, index=True)

    # Liker (can be user or expert)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), index=True)
    expert_id = Column(String, ForeignKey("experts.id", ondelete="CASCADE"), index=True)

    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    content = relationship("ExpertContentDB", back_populates="likes")
    user = relationship("UserDB")
    expert = relationship("ExpertDB")

    __table_args__ = (
        Index('idx_content_user_unique', 'content_id', 'user_id', unique=True, postgresql_where=text("user_id IS NOT NULL")),
        Index('idx_content_expert_unique', 'content_id', 'expert_id', unique=True, postgresql_where=text("expert_id IS NOT NULL")),
    )


class ExpertServiceRequestDB(Base):
    """Companies requesting expert services for specific tenders."""

    __tablename__ = "expert_service_requests"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    company_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    expert_id = Column(String, ForeignKey("experts.id", ondelete="CASCADE"), nullable=False, index=True)
    tender_id = Column(String, ForeignKey("tenders.id", ondelete="SET NULL"), index=True)

    # Request details
    request_type = Column(String)  # consultation/bid_writing/full_service/other
    description = Column(Text, nullable=False)
    budget_min = Column(Float)
    budget_max = Column(Float)
    deadline = Column(DateTime)

    # Status tracking
    status = Column(String, default='pending', index=True)  # pending/accepted/rejected/completed/cancelled

    # Expert response
    response_message = Column(Text)
    proposed_rate = Column(Float)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    responded_at = Column(DateTime)
    completed_at = Column(DateTime)

    # Relationships
    company = relationship("UserDB")
    expert = relationship("ExpertDB", back_populates="service_requests")
    tender = relationship("TenderDB")
    payments = relationship("ExpertPaymentDB", back_populates="request")


class ExpertApplicationDB(Base):
    """Experts applying to tenders proactively."""

    __tablename__ = "expert_applications"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    expert_id = Column(String, ForeignKey("experts.id", ondelete="CASCADE"), nullable=False, index=True)
    tender_id = Column(String, ForeignKey("tenders.id", ondelete="CASCADE"), nullable=False, index=True)
    company_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    # Application details
    cover_letter = Column(Text, nullable=False)
    proposed_rate = Column(Float)
    estimated_hours = Column(Integer)
    timeline = Column(String)

    # Supporting documents
    documents = Column(JSONB, default=list)  # [{"name": "...", "file_path": "..."}]

    # Status tracking
    status = Column(String, default='pending', index=True)  # pending/shortlisted/accepted/rejected

    # Company feedback
    company_notes = Column(Text)

    # Timestamps
    submitted_at = Column(DateTime, default=datetime.utcnow, index=True)
    reviewed_at = Column(DateTime)

    # Relationships
    expert = relationship("ExpertDB", back_populates="applications")
    tender = relationship("TenderDB")
    company = relationship("UserDB")

    __table_args__ = (
        Index('idx_expert_tender_unique', 'expert_id', 'tender_id', unique=True),
    )


class ExpertCollaborationDB(Base):
    """Experts working together on specific tenders."""

    __tablename__ = "expert_collaborations"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    tender_id = Column(String, ForeignKey("tenders.id", ondelete="CASCADE"), nullable=False, index=True)
    primary_expert_id = Column(String, ForeignKey("experts.id", ondelete="CASCADE"), nullable=False, index=True)
    collaborator_expert_id = Column(String, ForeignKey("experts.id", ondelete="CASCADE"), nullable=False, index=True)

    # Collaboration details
    role = Column(String)  # Description of collaborator's role
    share_percentage = Column(Float)  # Percentage of project earnings

    # Status tracking
    status = Column(String, default='invited', index=True)  # invited/accepted/declined/active/completed

    # Timestamps
    invited_at = Column(DateTime, default=datetime.utcnow)
    responded_at = Column(DateTime)

    # Relationships
    tender = relationship("TenderDB")
    primary_expert = relationship("ExpertDB", foreign_keys=[primary_expert_id])
    collaborator = relationship("ExpertDB", foreign_keys=[collaborator_expert_id])


class ExpertReviewDB(Base):
    """Company reviews of expert services."""

    __tablename__ = "expert_reviews"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    expert_id = Column(String, ForeignKey("experts.id", ondelete="CASCADE"), nullable=False, index=True)
    company_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    request_id = Column(String, ForeignKey("expert_service_requests.id", ondelete="SET NULL"), index=True)
    hiring_request_id = Column(String, ForeignKey("expert_hiring_requests.id", ondelete="SET NULL"), index=True)

    # Review data
    rating = Column(Integer, nullable=False)  # 1-10 scale (legacy 1-5)
    review_text = Column(Text)

    # Optional categories
    communication_rating = Column(Integer)  # 1-5
    quality_rating = Column(Integer)  # 1-5
    timeliness_rating = Column(Integer)  # 1-5
    professionalism_rating = Column(Integer)  # 1-5

    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    expert = relationship("ExpertDB", back_populates="reviews")
    company = relationship("UserDB")
    request = relationship("ExpertServiceRequestDB")

    __table_args__ = (
        Index('idx_expert_company_request_unique', 'expert_id', 'company_id', 'request_id', unique=True),
        Index('idx_expert_company_hiring_request_unique', 'expert_id', 'company_id', 'hiring_request_id', unique=True),
    )


class ExpertPaymentDB(Base):
    """Payment tracking and transactions for expert services."""

    __tablename__ = "expert_payments"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    expert_id = Column(String, ForeignKey("experts.id", ondelete="CASCADE"), nullable=False, index=True)
    company_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    request_id = Column(String, ForeignKey("expert_service_requests.id", ondelete="SET NULL"), index=True)

    # Payment details
    amount = Column(Float, nullable=False)
    currency = Column(String, default='INR')
    platform_fee_percentage = Column(Float, default=10.0)  # 10% platform fee
    platform_fee = Column(Float, nullable=False)
    expert_payout = Column(Float, nullable=False)

    # Payment processing
    payment_status = Column(String, default='pending', index=True)  # pending/processing/completed/failed/refunded
    payment_method = Column(String)  # card/bank_transfer/upi/wallet
    transaction_id = Column(String, index=True)  # External payment gateway reference

    # Invoice
    invoice_url = Column(String)  # Path to generated invoice PDF
    invoice_number = Column(String, unique=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    paid_at = Column(DateTime)
    payout_at = Column(DateTime)  # When expert received the money

    # Relationships
    expert = relationship("ExpertDB", back_populates="payments")
    company = relationship("UserDB")
    request = relationship("ExpertServiceRequestDB", back_populates="payments")


class ExpertFavoriteTenderDB(Base):
    """Tenders favorited/bookmarked by experts."""

    __tablename__ = "expert_favorite_tenders"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    expert_id = Column(String, ForeignKey("experts.id", ondelete="CASCADE"), nullable=False, index=True)
    tender_id = Column(String, ForeignKey("tenders.id", ondelete="CASCADE"), nullable=False, index=True)

    # Expert's private notes
    notes = Column(Text)

    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    expert = relationship("ExpertDB", back_populates="favorite_tenders")
    tender = relationship("TenderDB")

    __table_args__ = (
        Index('idx_expert_tender_favorite_unique', 'expert_id', 'tender_id', unique=True),
    )


class ExpertNotificationDB(Base):
    """Expert-specific notifications."""

    __tablename__ = "expert_notifications"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    expert_id = Column(String, ForeignKey("experts.id", ondelete="CASCADE"), nullable=False, index=True)

    # Notification details
    notification_type = Column(String, index=True)  # service_request/application_response/payment/collaboration/content_engagement
    title = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    link = Column(String)  # URL to relevant page

    # Status
    is_read = Column(Boolean, default=False, index=True)

    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationship
    expert = relationship("ExpertDB", back_populates="notifications")


class ExpertHiringRequestDB(Base):
    """Open expert hiring requests created by companies for projects."""
    
    __tablename__ = "expert_hiring_requests"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    company_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="SET NULL"), nullable=True)
    tender_id = Column(String, ForeignKey("tenders.id", ondelete="SET NULL"), nullable=True)
    
    # Request details
    request_name = Column(String, nullable=False)  # Name/title of request
    description = Column(Text, nullable=False)  # Detailed description
    budget_type = Column(String, nullable=False)  # "fixed" or "negotiable"
    budget_amount = Column(Float, nullable=True)  # Amount if fixed
    budget_min = Column(Float, nullable=True)  # Min if negotiable
    budget_max = Column(Float, nullable=True)  # Max if negotiable
    
    # Auto-populated company details
    company_name = Column(String, nullable=False)
    company_location = Column(String, nullable=True)
    
    # Auto-populated tender/project details
    tender_title = Column(String, nullable=True)
    tender_sector = Column(String, nullable=True)
    tender_state = Column(String, nullable=True)  # Legacy field, kept for backward compatibility
    tender_location = Column(String, nullable=True)  # Location from work_item_details
    
    # Status
    status = Column(String, default='open', index=True)  # open/closed/filled
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    closed_at = Column(DateTime, nullable=True)
    
    # Relationships
    company = relationship("UserDB")
    project = relationship("ProjectDB")
    tender = relationship("TenderDB")
    applications = relationship("ExpertHiringApplicationDB", back_populates="request", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_expert_hiring_status', 'status'),
        Index('idx_expert_hiring_company', 'company_id'),
        Index('idx_expert_hiring_created', 'created_at'),
    )


class ExpertHiringApplicationDB(Base):
    """Expert applications to hiring requests."""
    
    __tablename__ = "expert_hiring_applications"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    request_id = Column(String, ForeignKey("expert_hiring_requests.id", ondelete="CASCADE"), nullable=False, index=True)
    expert_id = Column(String, ForeignKey("experts.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Application details
    cover_letter = Column(Text, nullable=False)
    proposed_rate = Column(Float, nullable=True)
    estimated_timeline = Column(String, nullable=True)
    relevant_experience = Column(Text, nullable=True)
    
    # Status
    status = Column(String, default='pending', index=True)  # pending/shortlisted/accepted/rejected/completed
    
    # Manager notes
    manager_notes = Column(Text, nullable=True)
    
    # Timestamps
    applied_at = Column(DateTime, default=datetime.utcnow, index=True)
    reviewed_at = Column(DateTime, nullable=True)
    
    # Relationships
    request = relationship("ExpertHiringRequestDB", back_populates="applications")
    expert = relationship("ExpertDB")
    
    __table_args__ = (
        Index('idx_expert_app_request', 'request_id'),
        Index('idx_expert_app_expert', 'expert_id'),
        Index('idx_expert_app_status', 'status'),
        Index('idx_expert_app_unique', 'request_id', 'expert_id', unique=True),  # Prevent duplicate applications
    )


class ExpertProjectMessageDB(Base):
    """Chat messages for expert-led projects (hiring requests or collaborations)."""
    
    __tablename__ = "expert_project_messages"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    channel_type = Column(String, nullable=False)  # hiring or collaboration
    channel_id = Column(String, nullable=False, index=True)
    hiring_request_id = Column(String, ForeignKey("expert_hiring_requests.id", ondelete="SET NULL"), nullable=True)
    collaboration_id = Column(String, ForeignKey("expert_collaborations.id", ondelete="SET NULL"), nullable=True)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="SET NULL"), nullable=True)
    tender_id = Column(String, ForeignKey("tenders.id", ondelete="SET NULL"), nullable=True)
    sender_expert_id = Column(String, ForeignKey("experts.id", ondelete="CASCADE"), nullable=True, index=True)
    sender_user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)
    message = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    sender_expert = relationship("ExpertDB")
    sender_user = relationship("UserDB")
    hiring_request = relationship("ExpertHiringRequestDB")
    collaboration = relationship("ExpertCollaborationDB")
    project = relationship("ProjectDB")
    tender = relationship("TenderDB")
    
    __table_args__ = (
        Index('idx_project_channel', 'channel_type', 'channel_id'),
    )


class ExpertProjectTaskDB(Base):
    """Tasks assigned within expert project workspaces."""
    
    __tablename__ = "expert_project_tasks"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    channel_type = Column(String, nullable=False)
    channel_id = Column(String, nullable=False, index=True)
    hiring_request_id = Column(String, ForeignKey("expert_hiring_requests.id", ondelete="SET NULL"), nullable=True)
    collaboration_id = Column(String, ForeignKey("expert_collaborations.id", ondelete="SET NULL"), nullable=True)
    title = Column(String, nullable=False)
    description = Column(Text)
    priority = Column(String, default='medium')
    status = Column(String, default='pending')
    deadline = Column(DateTime, nullable=True)
    created_by_user_id = Column(String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_by_expert_id = Column(String, ForeignKey("experts.id", ondelete="SET NULL"), nullable=True)
    assignee_expert_id = Column(String, ForeignKey("experts.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    creator_user = relationship("UserDB", foreign_keys=[created_by_user_id])
    creator_expert = relationship("ExpertDB", foreign_keys=[created_by_expert_id])
    assignee_expert = relationship("ExpertDB", foreign_keys=[assignee_expert_id])
    hiring_request = relationship("ExpertHiringRequestDB")
    collaboration = relationship("ExpertCollaborationDB")
    
    __table_args__ = (
        Index('idx_project_tasks_channel', 'channel_type', 'channel_id'),
        Index('idx_project_tasks_status', 'status'),
    )


class ExpertProjectTaskFileDB(Base):
    """Files uploaded for expert project tasks."""
    
    __tablename__ = "expert_project_task_files"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    task_id = Column(String, ForeignKey("expert_project_tasks.id", ondelete="CASCADE"), nullable=False, index=True)
    channel_type = Column(String, nullable=False)
    channel_id = Column(String, nullable=False, index=True)
    
    # File details
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_size = Column(Integer)
    file_type = Column(String)
    description = Column(Text)
    
    # Uploader
    uploaded_by_expert_id = Column(String, ForeignKey("experts.id", ondelete="SET NULL"), nullable=True)
    uploaded_by_user_id = Column(String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    
    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    task = relationship("ExpertProjectTaskDB", backref="files")
    uploader_expert = relationship("ExpertDB", foreign_keys=[uploaded_by_expert_id])
    uploader_user = relationship("UserDB", foreign_keys=[uploaded_by_user_id])
    
    __table_args__ = (
        Index('idx_expert_task_files', 'task_id'),
    )


class ExpertProjectTaskProgressUpdateDB(Base):
    """Progress updates for expert project tasks."""
    
    __tablename__ = "expert_project_task_progress_updates"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    task_id = Column(String, ForeignKey("expert_project_tasks.id", ondelete="CASCADE"), nullable=False, index=True)
    channel_type = Column(String, nullable=False)
    channel_id = Column(String, nullable=False, index=True)
    
    # Update content
    update_text = Column(Text, nullable=False)
    
    # Author
    expert_id = Column(String, ForeignKey("experts.id", ondelete="CASCADE"), nullable=True, index=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)
    
    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    task = relationship("ExpertProjectTaskDB", backref="progress_updates")
    expert = relationship("ExpertDB", foreign_keys=[expert_id])
    user = relationship("UserDB", foreign_keys=[user_id])
    
    __table_args__ = (
        Index('idx_expert_task_progress', 'task_id', 'created_at'),
    )


class ExpertProjectTaskQueryDB(Base):
    """Queries/concerns raised by experts for project tasks."""
    
    __tablename__ = "expert_project_task_queries"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    task_id = Column(String, ForeignKey("expert_project_tasks.id", ondelete="CASCADE"), nullable=False, index=True)
    channel_type = Column(String, nullable=False)
    channel_id = Column(String, nullable=False, index=True)
    
    # Query details
    query_type = Column(String, nullable=False)  # clarification/timeline/resources/blocker/other
    title = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    priority = Column(String, default='medium')  # low/medium/high/urgent
    status = Column(String, default='open', index=True)  # open/in_progress/resolved/closed
    
    # Author
    raised_by_expert_id = Column(String, ForeignKey("experts.id", ondelete="CASCADE"), nullable=True, index=True)
    
    # Response
    responded_by_user_id = Column(String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    response = Column(Text)
    responded_at = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    task = relationship("ExpertProjectTaskDB", backref="queries")
    raised_by = relationship("ExpertDB", foreign_keys=[raised_by_expert_id])
    responded_by = relationship("UserDB", foreign_keys=[responded_by_user_id])
    
    __table_args__ = (
        Index('idx_expert_task_queries', 'task_id', 'status'),
    )


class EmployeePerformanceRatingDB(Base):
    """Weekly performance ratings given by managers to employees."""
    
    __tablename__ = "employee_performance_ratings"
    
    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(String, ForeignKey("employees.id", ondelete="CASCADE"), nullable=False, index=True)
    manager_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Rating details
    rating = Column(Float, nullable=False)  # 0-5 scale
    week_start_date = Column(Date, nullable=False, index=True)  # Monday of the week
    week_end_date = Column(Date, nullable=False)  # Sunday of the week
    
    # Optional feedback
    feedback = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    employee = relationship("EmployeeDB", backref="performance_ratings")
    manager = relationship("UserDB", backref="given_ratings")


class TenderAnalysisReportDB(Base):
    """
    SQLAlchemy model for employee tender analysis reports.
    Stores comprehensive analysis with structured sections and free-form notes.
    Only one report allowed per tender (owned by first creator).
    """
    __tablename__ = "tender_analysis_reports"

    id = Column(Integer, primary_key=True, index=True)
    tender_id = Column(String, ForeignKey("tenders.id", ondelete="CASCADE"), nullable=False, unique=True, index=True)
    employee_id = Column(String, ForeignKey("employees.id"), nullable=False)  # Report owner (first creator)

    # Report status
    status = Column(String, default="draft", nullable=False)  # draft, submitted

    # Structured sections (all TEXT fields for rich content)
    executive_summary = Column(Text)  # High-level overview
    eligibility_analysis = Column(Text)  # Company eligibility assessment
    technical_requirements = Column(Text)  # Technical specifications analysis
    financial_assessment = Column(Text)  # Cost estimates, budget analysis
    risk_assessment = Column(Text)  # Identified risks and mitigation
    compliance_review = Column(Text)  # Regulatory/compliance requirements
    recommendations = Column(Text)  # Final recommendations (bid/no-bid)

    # Free-form additional notes
    additional_notes = Column(Text)  # Open-ended analysis

    # Version tracking
    version = Column(Integer, default=1, nullable=False)
    edit_count = Column(Integer, default=0, nullable=False)
    last_edited_by = Column(String, ForeignKey("employees.id"))
    last_edited_at = Column(DateTime)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    submitted_at = Column(DateTime)  # When status changed from draft to submitted
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    tender = relationship("TenderDB", back_populates="analysis_report")
    owner = relationship("EmployeeDB", foreign_keys=[employee_id], back_populates="owned_reports")
    last_editor = relationship("EmployeeDB", foreign_keys=[last_edited_by], back_populates="edited_reports")
    attachments = relationship("ReportAttachmentDB", back_populates="report", cascade="all, delete-orphan")

    # Indexes for query optimization
    __table_args__ = (
        Index('idx_analysis_tender_employee', 'tender_id', 'employee_id'),
        Index('idx_analysis_status', 'status'),
        Index('idx_analysis_created', 'created_at'),
    )


class ReportAttachmentDB(Base):
    """
    SQLAlchemy model for tender analysis report file attachments.
    Stores supporting documents (PDFs, Excel, presentations, images).
    """
    __tablename__ = "report_attachments"

    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(Integer, ForeignKey("tender_analysis_reports.id", ondelete="CASCADE"), nullable=False)
    employee_id = Column(String, ForeignKey("employees.id"), nullable=False)  # Who uploaded

    # File metadata
    filename = Column(String, nullable=False)
    mime_type = Column(String, nullable=False)  # application/pdf, application/vnd.ms-excel, etc.
    file_size = Column(Integer, nullable=False)  # Bytes
    file_data = Column(LargeBinary, nullable=False)  # Binary storage (similar to TaskFileDB)

    # Optional description
    description = Column(String)

    # Timestamp
    uploaded_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    report = relationship("TenderAnalysisReportDB", back_populates="attachments")
    uploader = relationship("EmployeeDB", back_populates="report_attachments")

    # Indexes
    __table_args__ = (
        Index('idx_attachment_report', 'report_id'),
    )


class TenderAIInsightsDB(Base):
    """
    SQLAlchemy model for AI-extracted PQ and Eligibility criteria from tender documents.
    Uses GPT-4o to extract structured criteria from tender PDFs.
    """
    __tablename__ = "tender_ai_insights"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    tender_id = Column(String, ForeignKey("tenders.id", ondelete="CASCADE"), nullable=False, index=True)
    document_id = Column(Integer, ForeignKey("tender_documents.id"), nullable=True)  # Source document

    # Extraction metadata
    extraction_status = Column(String, default="not_started", nullable=False)  # not_started, processing, completed, failed
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)

    # Document info
    document_filename = Column(String, nullable=True)
    document_page_count = Column(Integer, nullable=True)

    # Extracted data (JSONB format for PostgreSQL)
    pq_criteria = Column(JSONB, nullable=True, server_default='[]')  # List of PQ criteria objects
    eligibility_criteria = Column(JSONB, nullable=True, server_default='[]')  # List of eligibility criteria objects
    sections = Column(JSONB, nullable=True, server_default='[]')  # Extracted sections
    summary = Column(JSONB, nullable=True, server_default='{}')  # Summary statistics
    raw_gpt_response = Column(Text, nullable=True)  # Raw verbatim GPT-4o response for transparency

    # Statistics (denormalized for quick access)
    total_pq_criteria = Column(Integer, default=0, nullable=False)
    total_eligibility_criteria = Column(Integer, default=0, nullable=False)
    mandatory_count = Column(Integer, default=0, nullable=False)
    optional_count = Column(Integer, default=0, nullable=False)

    # Extraction quality metadata (Phase 3 enhancement)
    extraction_warnings = Column(JSONB, nullable=True, server_default='[]')  # List of warning messages
    text_length = Column(Integer, nullable=True)  # Extracted text length in chars
    was_truncated = Column(Boolean, default=False, nullable=False)  # Flag if document was truncated

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    tender = relationship("TenderDB", back_populates="ai_insights")

    # Indexes for query optimization
    __table_args__ = (
        Index('idx_ai_insights_tender', 'tender_id'),
        Index('idx_ai_insights_status', 'extraction_status'),
        Index('idx_ai_insights_created', 'created_at'),
    )


class TenderEmbeddingDB(Base):
    """
    Cache for tender and user profile embeddings using OpenAI text-embedding-3-large.
    Stores embeddings as binary data for fast semantic similarity calculations.
    """
    __tablename__ = "tender_embeddings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    text_hash = Column(String(64), unique=True, index=True, nullable=False)  # SHA-256 hash of source text
    text_type = Column(String(20), nullable=False)  # 'tender' or 'profile'
    embedding = Column(LargeBinary, nullable=False)  # numpy array as bytes (3072 float32 values)
    model = Column(String(50), nullable=False, default='text-embedding-3-large')  # OpenAI model used
    dimensions = Column(Integer, nullable=False, default=3072)  # Embedding dimensions

    # Cache statistics
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_accessed = Column(DateTime, default=datetime.utcnow, nullable=False)
    access_count = Column(Integer, default=1, nullable=False)

    # Indexes for performance
    __table_args__ = (
        Index('idx_text_hash_type', 'text_hash', 'text_type'),
        Index('idx_created_at', 'created_at'),
        Index('idx_last_accessed', 'last_accessed'),
    )


def get_db():
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def ensure_company_schema():
    """Ensure legacy databases have new company detail columns."""
    try:
        with engine.begin() as conn:
            inspector = inspect(conn)
            if "companies" not in inspector.get_table_names():
                return

            existing_columns = {col["name"] for col in inspector.get_columns("companies")}
            alterations = []

            if "legal_details" not in existing_columns:
                alterations.append("ADD COLUMN legal_details JSONB DEFAULT '{}'::jsonb")
            if "financial_details" not in existing_columns:
                alterations.append("ADD COLUMN financial_details JSONB DEFAULT '{}'::jsonb")

            for statement in alterations:
                logger.info(f"Altering companies table: {statement}")
                conn.execute(text(f"ALTER TABLE companies {statement}"))

            # Initialize newly added JSON columns with empty JSON objects
            if "legal_details" not in existing_columns or "financial_details" not in existing_columns:
                conn.execute(
                    text(
                        "UPDATE companies "
                        "SET legal_details = COALESCE(legal_details, '{}'::jsonb), "
                        "financial_details = COALESCE(financial_details, '{}'::jsonb)"
                    )
                )

    except Exception as exc:
        logger.warning(f"Could not ensure company schema: {exc}")


def ensure_certificate_schema():
    """Ensure certificates table has the latest extraction columns."""
    try:
        with engine.begin() as conn:
            inspector = inspect(conn)
            if "certificates" not in inspector.get_table_names():
                return

            existing_columns = {col["name"] for col in inspector.get_columns("certificates")}
            backend = database_url.get_backend_name()

            json_array_type = "JSONB DEFAULT '[]'::jsonb" if backend == "postgresql" else "TEXT DEFAULT '[]'"
            json_object_type = "JSONB" if backend == "postgresql" else "TEXT"
            text_type = "TEXT"
            timestamp_type = "TIMESTAMP" if backend == "postgresql" else "DATETIME"
            float_type = "DOUBLE PRECISION" if backend == "postgresql" else "REAL"

            additions = [
                ("sectors", json_array_type),
                ("sub_sectors", json_array_type),
                ("consultancy_fee_inr", text_type),
                ("project_value_inr", text_type),
                ("scope_of_work", text_type),
                ("start_date", timestamp_type),
                ("end_date", timestamp_type),
                ("duration", text_type),
                ("issuing_authority_details", text_type),
                ("performance_remarks", text_type),
                ("certificate_number", text_type),
                ("signing_authority_details", text_type),
                ("role_lead_jv", text_type),
                ("jv_partners", json_array_type),
                ("funding_agency", text_type),
                ("confidence_score", float_type),
                ("metrics", json_object_type),
                ("verbatim_certificate", text_type),
                ("s3_key", text_type),  # S3 object key for certificate storage
                ("s3_url", text_type),  # S3 URL for certificate access
            ]

            for column_name, column_def in additions:
                if column_name in existing_columns:
                    continue

                if backend == "postgresql":
                    statement = f"ALTER TABLE certificates ADD COLUMN IF NOT EXISTS {column_name} {column_def}"
                else:
                    statement = f"ALTER TABLE certificates ADD COLUMN {column_name} {column_def}"

                logger.info(f"Altering certificates table: {statement}")
                conn.execute(text(statement))

    except Exception as exc:
        logger.warning(f"Could not ensure certificate schema: {exc}")


def ensure_project_schema():
    """Ensure projects table includes required columns."""
    try:
        with engine.begin() as conn:
            inspector = inspect(conn)
            if "projects" not in inspector.get_table_names():
                return

            existing_columns = {col["name"] for col in inspector.get_columns("projects")}

            if "financing_authority" not in existing_columns:
                logger.info("Altering projects table: ADD COLUMN financing_authority TEXT")
                conn.execute(text("ALTER TABLE projects ADD COLUMN financing_authority TEXT"))

            logger.info("Backfilling financing_authority for existing projects")
            conn.execute(text(
                "UPDATE projects "
                "SET financing_authority = 'Financing Not Required' "
                "WHERE financing_authority IS NULL OR TRIM(financing_authority) = ''"
            ))
    except Exception as exc:
        logger.warning(f"Could not ensure project schema: {exc}")


def create_tables():
    """Create all database tables."""
    Base.metadata.create_all(bind=engine)
    ensure_company_schema()
    ensure_project_schema()
    ensure_certificate_schema()


def cleanup_old_tenders(db: Session, days_old: int = 60) -> int:
    """Remove tenders older than specified days (skipping favorites)."""

    cutoff_date = datetime.utcnow() - timedelta(days=days_old)

    old_tenders = db.query(TenderDB).filter(
        TenderDB.published_at < cutoff_date,
        ~TenderDB.favorites.any()
    ).all()

    count = len(old_tenders)

    for tender in old_tenders:
        db.delete(tender)

    db.commit()
    logger.info(f"Cleaned up {count} old tenders")
    return count


def cleanup_expired_tenders(db: Session) -> int:
    """
    Remove tenders with expired deadlines that have not been awarded.

    This function deletes tenders where:
    - deadline has passed (deadline < now)
    - tender has not been awarded (awarded = False)

    Before deletion it:
    - Creates a `tender_expired` notification for every affected favorite/shortlist user
    - Clears any pending deadline_* notifications for those users

    Favorites, shortlisted entries, and related records are removed via cascade.

    Returns:
        int: Number of expired tenders deleted
    """
    now = datetime.utcnow()

    expired_tenders = db.query(TenderDB).filter(
        TenderDB.deadline.isnot(None),
        TenderDB.deadline < now,
        TenderDB.awarded == False
    ).all()

    if not expired_tenders:
        return 0

    notifications_created = 0

    for tender in expired_tenders:
        tender_identifier = tender.title or tender.tender_reference_number or tender.id

        # Collect impacted favorites and shortlisted entries before cascade removal
        favorites = list(tender.favorites)
        shortlisted = db.query(ShortlistedTenderDB).filter(
            ShortlistedTenderDB.tender_id == tender.id
        ).all()

        impacted_users = set()

        for fav in favorites:
            impacted_users.add(fav.user_id)
            db.query(NotificationDB).filter(
                NotificationDB.user_id == fav.user_id,
                NotificationDB.tender_id == tender.id,
                NotificationDB.notification_type.like('deadline_%')
            ).delete(synchronize_session=False)

        shortlist_ids = []

        for shortlist in shortlisted:
            impacted_users.add(shortlist.user_id)
            db.query(NotificationDB).filter(
                NotificationDB.user_id == shortlist.user_id,
                NotificationDB.tender_id == tender.id,
                NotificationDB.notification_type.like('deadline_%')
            ).delete(synchronize_session=False)
            shortlist_ids.append(shortlist.id)

        if shortlist_ids:
            db.query(StageDocumentDB).filter(StageDocumentDB.shortlist_id.in_(shortlist_ids)).delete(synchronize_session=False)
            db.query(ShortlistedTenderDB).filter(ShortlistedTenderDB.id.in_(shortlist_ids)).delete(synchronize_session=False)

        for user_id in impacted_users:
            existing = db.query(NotificationDB).filter(
                NotificationDB.user_id == user_id,
                NotificationDB.tender_id == tender.id,
                NotificationDB.notification_type == 'tender_expired'
            ).first()

            if not existing:
                notification = NotificationDB(
                    user_id=user_id,
                    tender_id=tender.id,
                    notification_type='tender_expired',
                    message=f"Tender expired and was removed: {tender_identifier}",
                    tender_title=tender.title,
                    days_remaining=-1,
                    is_read=False
                )
                db.add(notification)
                notifications_created += 1

        # Mark calendar activities as source_deleted before deleting tender
        mark_activities_source_deleted(db, tender_id=tender.id)

        db.delete(tender)

    db.commit()
    logger.info(
        "Cleaned up %s expired non-awarded tenders (created %s expiration notifications)",
        len(expired_tenders),
        notifications_created,
    )
    return len(expired_tenders)


def cleanup_orphaned_records(db: Session) -> Dict[str, int]:
    """
    Remove orphaned records where the referenced tender no longer exists.

    This cleans up:
    - Shortlisted tenders
    - Rejected tenders
    - Notifications
    - Reminders
    - Dumped tenders
    - AI chat messages
    - Tender responses
    - Tender assignments

    Returns:
        Dict[str, int]: Count of deleted records per table
    """
    counts = {}

    # Clean up shortlisted tenders
    orphaned_shortlisted = db.query(ShortlistedTenderDB).filter(
        ~db.query(TenderDB).filter(TenderDB.id == ShortlistedTenderDB.tender_id).exists()
    ).all()
    counts['shortlisted_tenders'] = len(orphaned_shortlisted)
    for record in orphaned_shortlisted:
        db.delete(record)

    # Clean up stage documents without a shortlist
    orphaned_stage_docs = db.query(StageDocumentDB).filter(
        ~db.query(ShortlistedTenderDB).filter(ShortlistedTenderDB.id == StageDocumentDB.shortlist_id).exists()
    ).all()
    counts['stage_documents'] = len(orphaned_stage_docs)
    for record in orphaned_stage_docs:
        db.delete(record)

    # Clean up rejected tenders
    orphaned_rejected = db.query(RejectedTenderDB).filter(
        ~db.query(TenderDB).filter(TenderDB.id == RejectedTenderDB.tender_id).exists()
    ).all()
    counts['rejected_tenders'] = len(orphaned_rejected)
    for record in orphaned_rejected:
        db.delete(record)

    # Clean up notifications
    orphaned_notifications = db.query(NotificationDB).filter(
        ~db.query(TenderDB).filter(TenderDB.id == NotificationDB.tender_id).exists()
    ).all()
    counts['notifications'] = len(orphaned_notifications)
    for record in orphaned_notifications:
        db.delete(record)

    # Clean up reminders
    orphaned_reminders = db.query(ReminderDB).filter(
        ~db.query(TenderDB).filter(TenderDB.id == ReminderDB.tender_id).exists()
    ).all()
    counts['reminders'] = len(orphaned_reminders)
    for record in orphaned_reminders:
        db.delete(record)

    # Clean up dumped tenders
    orphaned_dumped = db.query(DumpedTenderDB).filter(
        ~db.query(TenderDB).filter(TenderDB.id == DumpedTenderDB.tender_id).exists()
    ).all()
    counts['dumped_tenders'] = len(orphaned_dumped)
    for record in orphaned_dumped:
        db.delete(record)

    # Clean up tender responses
    orphaned_responses = db.query(TenderResponseDB).filter(
        ~db.query(TenderDB).filter(TenderDB.id == TenderResponseDB.tender_id).exists()
    ).all()
    counts['tender_responses'] = len(orphaned_responses)
    for record in orphaned_responses:
        db.delete(record)

    # Clean up tender assignments
    orphaned_assignments = db.query(TenderAssignmentDB).filter(
        ~db.query(TenderDB).filter(TenderDB.id == TenderAssignmentDB.tender_id).exists()
    ).all()
    counts['tender_assignments'] = len(orphaned_assignments)
    for record in orphaned_assignments:
        db.delete(record)

    db.commit()

    total = sum(counts.values())
    if total > 0:
        logger.info(f"Cleaned up orphaned records: {counts}")

    return counts


def save_calendar_activity(
    db: Session,
    user_id: str,
    activity_date: datetime,
    activity_type: str,
    title: str,
    description: str,
    tender_id: Optional[str] = None,
    project_id: Optional[int] = None,
    reminder_id: Optional[int] = None
) -> Optional[CalendarActivityDB]:
    """
    Save a calendar activity to persistent storage.

    Args:
        db: Database session
        user_id: User ID
        activity_date: Date of the activity
        activity_type: Type of activity (deadline, activity, reminder)
        title: Activity title
        description: Activity description
        tender_id: Optional tender ID
        project_id: Optional project ID
        reminder_id: Optional reminder ID

    Returns:
        CalendarActivityDB instance if successful, None otherwise
    """
    try:
        # Check if activity already exists
        existing = db.query(CalendarActivityDB).filter(
            CalendarActivityDB.user_id == user_id,
            CalendarActivityDB.activity_date == activity_date,
            CalendarActivityDB.activity_type == activity_type,
            CalendarActivityDB.title == title
        ).first()

        if existing:
            logger.debug(f"Calendar activity already exists: {title}")
            return existing

        # Create new activity
        activity = CalendarActivityDB(
            user_id=user_id,
            activity_date=activity_date,
            activity_type=activity_type,
            title=title,
            description=description,
            tender_id=tender_id,
            project_id=project_id,
            reminder_id=reminder_id,
            source_deleted=False,
            is_active=True
        )

        db.add(activity)
        db.commit()
        db.refresh(activity)

        logger.info(f"Saved calendar activity: {title} for {activity_date.strftime('%Y-%m-%d')}")
        return activity

    except Exception as e:
        logger.error(f"Error saving calendar activity: {str(e)}")
        db.rollback()
        return None


def mark_activities_source_deleted(
    db: Session,
    tender_id: Optional[str] = None,
    project_id: Optional[int] = None,
    reminder_id: Optional[int] = None
) -> int:
    """
    Mark calendar activities as source_deleted when their source is deleted.

    Args:
        db: Database session
        tender_id: Tender ID if tender deleted
        project_id: Project ID if project deleted
        reminder_id: Reminder ID if reminder deleted

    Returns:
        Number of activities marked as deleted
    """
    try:
        query = db.query(CalendarActivityDB)

        if tender_id:
            query = query.filter(CalendarActivityDB.tender_id == tender_id)
        elif project_id:
            query = query.filter(CalendarActivityDB.project_id == project_id)
        elif reminder_id:
            query = query.filter(CalendarActivityDB.reminder_id == reminder_id)
        else:
            return 0

        activities = query.all()
        count = len(activities)

        for activity in activities:
            activity.source_deleted = True
            activity.deleted_at = datetime.utcnow()

        db.commit()

        if count > 0:
            logger.info(f"Marked {count} calendar activities as source_deleted")

        return count

    except Exception as e:
        logger.error(f"Error marking activities as deleted: {str(e)}")
        db.rollback()
        return 0


def parse_date_string(date_str: str) -> Optional[datetime]:
    """Parse partial tender date strings into datetime."""

    if not date_str or str(date_str).upper() == 'NA':
        return None

    formats = [
        "%d-%b-%Y %I:%M %p",
        "%d-%m-%Y %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y",
        "%Y-%m-%d",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except (ValueError, AttributeError):
            continue

    try:
        import re

        date_match = re.search(r'(\d{1,2})-(\w{3})-(\d{4})', str(date_str))
        if date_match:
            day, month_abbr, year = date_match.groups()
            month_map = {
                'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
                'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
                'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
            }
            if month_abbr in month_map:
                return datetime(int(year), month_map[month_abbr], int(day))
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.debug(f"Fallback date parsing failed for '{date_str}': {exc}")

    logger.warning(f"Could not parse date string: {date_str}")
    return None


def extract_numeric_value(value_str: Any) -> Optional[float]:
    """Convert strings like '13,21,807' or '₹ 5,00,000.00' into floats."""

    if value_str is None:
        return None

    raw = str(value_str).strip()
    if not raw or raw.upper() == 'NA':
        return None

    import re

    cleaned = re.sub(r'[^0-9.]', '', raw)
    if not cleaned:
        return None

    try:
        return float(cleaned)
    except ValueError:
        logger.warning(f"Could not parse numeric value: {value_str}")
        return None


def extract_state_from_org_chain(org_chain: str) -> str:
    """Best-effort extraction of the state from the organisation chain."""

    if not org_chain:
        return "Unknown"

    org_lower = org_chain.lower()
    state_patterns = {
        'rajasthan': 'Rajasthan',
        'maharashtra': 'Maharashtra',
        'gujarat': 'Gujarat',
        'karnataka': 'Karnataka',
        'tamil nadu': 'Tamil Nadu',
        'kerala': 'Kerala',
        'andhra pradesh': 'Andhra Pradesh',
        'telangana': 'Telangana',
        'west bengal': 'West Bengal',
        'uttar pradesh': 'Uttar Pradesh',
        'madhya pradesh': 'Madhya Pradesh',
        'bihar': 'Bihar',
        'odisha': 'Odisha',
        'punjab': 'Punjab',
        'haryana': 'Haryana',
        'himachal pradesh': 'Himachal Pradesh',
        'uttarakhand': 'Uttarakhand',
        'jharkhand': 'Jharkhand',
        'chhattisgarh': 'Chhattisgarh',
        'assam': 'Assam',
        'delhi': 'Delhi'
    }

    for pattern, state in state_patterns.items():
        if pattern in org_lower:
            return state

    return "Unknown"


def generate_tags(tender_data: Dict[str, Any]) -> List[str]:
    """Create lightweight tag list for quick filtering/search."""

    tags: List[str] = []

    category = tender_data.get('Tender Category')
    if category:
        tags.append(category)

    tender_type = tender_data.get('Tender Type')
    if tender_type:
        tags.append(tender_type)

    authority_info = tender_data.get('Tender Inviting Authority')
    if isinstance(authority_info, dict):
        authority_name = authority_info.get('Name', '')
        if authority_name:
            lowered = authority_name.lower()
            if 'forest' in lowered:
                tags.append('Forest Department')
            elif 'public works' in lowered or 'pwd' in lowered:
                tags.append('Public Works')
            elif 'medical' in lowered or 'hospital' in lowered:
                tags.append('Healthcare')
            elif 'education' in lowered or 'school' in lowered:
                tags.append('Education')

    work_details = tender_data.get('Work Item Details')
    if isinstance(work_details, dict):
        tender_value = extract_numeric_value(work_details.get('Tender Value (INR)', ''))
        if tender_value:
            if tender_value >= 10_000_000:
                tags.append('High Value')
            elif tender_value >= 1_000_000:
                tags.append('Medium Value')
            else:
                tags.append('Low Value')

    org_chain = tender_data.get('Organisation Chain', '')
    state = extract_state_from_org_chain(org_chain)
    if state != 'Unknown':
        tags.append(state)

    return list(dict.fromkeys(tags))


def clean_string_field(value: Any) -> str:
    """Collapse whitespace in free-form text fields."""

    if not value:
        return ""

    return " ".join(str(value).split())


def create_tender_from_scraper_data(tender_data: Dict[str, Any]) -> TenderDB:
    """Transform scraped tender payload into a TenderDB instance."""

    import uuid

    work_details = tender_data.get('Work Item Details', {})
    critical_dates = tender_data.get('Critical Dates', {})
    authority_info = tender_data.get('Tender Inviting Authority', {})

    title = clean_string_field(work_details.get('Title', '')) if isinstance(work_details, dict) else ''
    authority = clean_string_field(authority_info.get('Name', '')) if isinstance(authority_info, dict) else ''
    state = extract_state_from_org_chain(tender_data.get('Organisation Chain', ''))
    category = tender_data.get('Tender Category', '')

    deadline = None
    published_at = None
    if isinstance(critical_dates, dict):
        deadline = parse_date_string(critical_dates.get('Bid Submission End Date', ''))
        published_at = parse_date_string(critical_dates.get('Published Date', ''))

    estimated_value = None
    if isinstance(work_details, dict):
        estimated_value = extract_numeric_value(work_details.get('Tender Value (INR)', ''))

    summary = work_details.get('Work Description', '') if isinstance(work_details, dict) else ''
    if not summary:
        summary = title

    tags = generate_tags(tender_data)

    document_url = tender_data.get('scraping_metadata', {}).get('document_url', '')

    tender = TenderDB(
        id=str(uuid.uuid4()),
        organisation_chain=tender_data.get('Organisation Chain', ''),
        tender_reference_number=tender_data.get('Tender Reference Number', ''),
        tender_id=tender_data.get('Tender ID', ''),
        tender_type=tender_data.get('Tender Type', ''),
        tender_category=category,
        general_technical_evaluation_allowed=tender_data.get('General Technical Evaluation Allowed', ''),
        payment_mode=tender_data.get('Payment Mode', ''),
        withdrawal_allowed=tender_data.get('Withdrawal Allowed', ''),
        form_of_contract=tender_data.get('Form Of Contract', ''),
        no_of_covers=tender_data.get('No. of Covers', ''),
        itemwise_technical_evaluation_allowed=tender_data.get('ItemWise Technical Evaluation Allowed', ''),
        is_multi_currency_allowed_for_boq=tender_data.get('Is Multi Currency Allowed For BOQ', ''),
        is_multi_currency_allowed_for_fee=tender_data.get('Is Multi Currency Allowed For Fee', ''),
        allow_two_stage_bidding=tender_data.get('Allow Two Stage Bidding', ''),
        payment_instruments=tender_data.get('Payment Instruments', {}),
        covers_information=tender_data.get('Covers Information', []),
        tender_fee_details=tender_data.get('Tender Fee Details', {}),
        emd_fee_details=tender_data.get('EMD Fee Details', {}),
        work_item_details=work_details if isinstance(work_details, dict) else {},
        critical_dates=critical_dates if isinstance(critical_dates, dict) else {},
        tender_documents=tender_data.get('Tender Documents', {}),
        tender_inviting_authority=authority_info if isinstance(authority_info, dict) else {},
        additional_fields=tender_data.get('additional_fields', {}),
        scraped_at=datetime.utcnow(),
        source_url=tender_data.get('scraping_metadata', {}).get('source_url', ''),
        source=tender_data.get('scraping_metadata', {}).get('source', ''),
        scraper_version=tender_data.get('scraping_metadata', {}).get('scraper_version', ''),
        search_term_used=tender_data.get('search_term_used', ''),
        title=title,
        authority=authority,
        state=state,
        category=category,
        estimated_value=estimated_value,
        currency='INR',
        deadline=deadline,
        published_at=published_at,
        summary=summary,
        pdf_url=document_url,
        tags=tags,
    )

    return tender


def update_tender_with_openai_data(
    db: Session,
    record_id: str,
    tender_data: Dict[str, Any]
) -> bool:
    """
    Update an existing partial tender record with Open AI extracted data.

    Args:
        db: Database session
        record_id: UUID of the existing partial tender record
        tender_data: Complete tender data from OpenAI extraction

    Returns:
        bool: True if update successful, False otherwise
    """
    try:
        # Validate tender_data has actual content
        if not tender_data or not isinstance(tender_data, dict):
            logger.error(f"❌ tender_data is empty or invalid type: {type(tender_data)}")
            return False
        
        if len(tender_data) < 5:
            logger.error(f"❌ tender_data is too small (only {len(tender_data)} fields): {list(tender_data.keys())}")
            return False
        
        # Check for critical fields
        required_fields = ['Work Item Details', 'Critical Dates', 'Tender ID']
        missing_fields = [f for f in required_fields if f not in tender_data]
        if missing_fields:
            logger.warning(f"⚠️  Missing required fields in tender_data: {missing_fields}")
            logger.warning(f"Available fields: {list(tender_data.keys())}")
        
        # Query the existing record
        tender = db.query(TenderDB).filter(TenderDB.id == record_id).first()

        if not tender:
            logger.warning(f"No existing tender record found with id: {record_id}")
            return False

        logger.info(f"Updating existing tender record: {record_id}")
        logger.debug(f"Tender data fields: {list(tender_data.keys())}")

        # Extract data same way as create_tender_from_scraper_data
        work_details = tender_data.get('Work Item Details', {})
        critical_dates = tender_data.get('Critical Dates', {})
        authority_info = tender_data.get('Tender Inviting Authority', {})

        title = clean_string_field(work_details.get('Title', '')) if isinstance(work_details, dict) else ''
        authority = clean_string_field(authority_info.get('Name', '')) if isinstance(authority_info, dict) else ''
        state = extract_state_from_org_chain(tender_data.get('Organisation Chain', ''))
        category = tender_data.get('Tender Category', '')

        deadline = None
        published_at = None
        if isinstance(critical_dates, dict):
            deadline = parse_date_string(critical_dates.get('Bid Submission End Date', ''))
            published_at = parse_date_string(critical_dates.get('Published Date', ''))

        estimated_value = None
        if isinstance(work_details, dict):
            estimated_value = extract_numeric_value(work_details.get('Tender Value (INR)', ''))

        summary = work_details.get('Work Description', '') if isinstance(work_details, dict) else ''
        if not summary:
            summary = title

        tags = generate_tags(tender_data)

        document_url = tender_data.get('scraping_metadata', {}).get('document_url', '')

        # Update all fields with OpenAI extracted data
        tender.organisation_chain = tender_data.get('Organisation Chain', '')
        tender.tender_reference_number = tender_data.get('Tender Reference Number', '')
        tender.tender_id = tender_data.get('Tender ID', '')  # Update from temp_id to actual
        tender.tender_type = tender_data.get('Tender Type', '')
        tender.tender_category = category
        tender.general_technical_evaluation_allowed = tender_data.get('General Technical Evaluation Allowed', '')
        tender.payment_mode = tender_data.get('Payment Mode', '')
        tender.withdrawal_allowed = tender_data.get('Withdrawal Allowed', '')
        tender.form_of_contract = tender_data.get('Form Of Contract', '')
        tender.no_of_covers = tender_data.get('No. of Covers', '')
        tender.itemwise_technical_evaluation_allowed = tender_data.get('ItemWise Technical Evaluation Allowed', '')
        tender.is_multi_currency_allowed_for_boq = tender_data.get('Is Multi Currency Allowed For BOQ', '')
        tender.is_multi_currency_allowed_for_fee = tender_data.get('Is Multi Currency Allowed For Fee', '')
        tender.allow_two_stage_bidding = tender_data.get('Allow Two Stage Bidding', '')
        tender.payment_instruments = tender_data.get('Payment Instruments', {})
        tender.covers_information = tender_data.get('Covers Information', [])
        tender.tender_fee_details = tender_data.get('Tender Fee Details', {})
        tender.emd_fee_details = tender_data.get('EMD Fee Details', {})
        tender.work_item_details = work_details if isinstance(work_details, dict) else {}
        tender.critical_dates = critical_dates if isinstance(critical_dates, dict) else {}
        tender.tender_documents = tender_data.get('Tender Documents', {})
        tender.tender_inviting_authority = authority_info if isinstance(authority_info, dict) else {}

        # Update additional_fields but preserve file paths
        existing_additional = tender.additional_fields or {}
        new_additional = tender_data.get('additional_fields', {})
        # Merge: keep file paths from Phase 1, add new data from Phase 2
        existing_additional.update(new_additional)
        existing_additional['processing_status'] = 'completed'  # Mark as completed
        tender.additional_fields = existing_additional

        # Update metadata
        tender.scraper_version = tender_data.get('scraping_metadata', {}).get('scraper_version', '')
        tender.search_term_used = tender_data.get('search_term_used', '')

        # Update computed fields
        tender.title = title
        tender.authority = authority
        tender.state = state
        tender.category = category
        tender.estimated_value = estimated_value
        tender.deadline = deadline
        tender.published_at = published_at
        tender.summary = summary
        tender.pdf_url = document_url if document_url else tender.pdf_url  # Keep existing if no new URL
        tender.tags = tags
        tender.updated_at = datetime.utcnow()

        db.commit()
        logger.info(f"Successfully updated tender record: {record_id}")
        return True

    except Exception as e:
        logger.error(f"Error updating tender record {record_id}: {str(e)}")
        db.rollback()
        return False
