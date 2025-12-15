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

        # Check for analysis report
        analysis_report = db.query(TenderAnalysisReportDB).filter(
            TenderAnalysisReportDB.tender_id == assignment.tender.id
        ).first()

        has_analysis = analysis_report is not None
        is_analysis_owner = False
        if analysis_report:
            is_analysis_owner = (analysis_report.employee_id == current_employee.id)

        tender_summaries.append({
            'id': assignment.tender.id,
            'title': assignment.tender.title,
            'tasks_left': total_tasks - completed_tasks,
            'total_tasks': total_tasks,
            'nearest_deadline': nearest_deadline,
            'priority': assignment.priority,
            'role': assignment.role,
            'has_analysis': has_analysis,
            'is_analysis_owner': is_analysis_owner
        })

    return templates.TemplateResponse("employee_dashboard.html", {
        "request": request,
        "current_employee": current_employee,
        "tender_summaries": tender_summaries,
        "active_tenders": active_tenders,
        "pending_tasks": pending_tasks,
        "overdue_tasks": overdue_tasks,
        "due_today": due_today,
        "now": datetime.utcnow()
    })


@router.get("/employee/tender/{tender_id}", response_class=HTMLResponse)
async def employee_tender_detail(request: Request, tender_id: str, db: Session = Depends(get_db)):
    """Employee tender detail page with tasks."""
    current_employee = get_current_employee(request, db)
    if not current_employee:
        return RedirectResponse(url="/employee/login", status_code=302)

    # Get assignment for this employee and tender
    assignment = db.query(TenderAssignmentDB).filter(
        and_(TenderAssignmentDB.tender_id == tender_id, TenderAssignmentDB.employee_id == current_employee.id)
    ).first()

    if not assignment or not assignment.tender:
        raise HTTPException(status_code=404, detail="Tender assignment not found")

    # Get tasks for this assignment
    tasks = db.query(TaskDB).filter(TaskDB.assignment_id == assignment.id).all()

    task_comment_payloads = {}
    task_comment_counts = {}
    for task in tasks:
        comment_payload = []
        if task.comments:
            for comment in task.comments:
                author_name = None
                if comment.employee:
                    author_name = comment.employee.name
                comment_payload.append({
                    "id": comment.id,
                    "author": author_name,
                    "body": comment.comment,
                    "timestamp": comment.created_at.isoformat() if comment.created_at else None
                })
        task_comment_payloads[task.id] = comment_payload
        task_comment_counts[task.id] = len(comment_payload)

    # Get team members (other employees assigned to this tender)
    team_assignments = db.query(TenderAssignmentDB).filter(
        and_(TenderAssignmentDB.tender_id == tender_id, TenderAssignmentDB.employee_id != current_employee.id)
    ).all()
    team_members = [ta.employee for ta in team_assignments if ta.employee]

    # Get manager (user who assigned the tender)
    manager = None
    if assignment.assigned_by: # type: ignore
        manager = db.query(UserDB).filter(UserDB.id == assignment.assigned_by).first()
        if manager:
            # Create a mock employee object for the manager to include in chat
            from types import SimpleNamespace
            manager_employee = SimpleNamespace()
            manager_employee.id = f"manager_{manager.id}"
            manager_employee.name = f"{manager.name} (Manager)"
            manager_employee.email = manager.email
            team_members.append(manager_employee)

    # Get messages for this assignment with robust ordering
    messages = db.query(TenderMessageDB).filter(
        TenderMessageDB.assignment_id == assignment.id
    ).order_by(TenderMessageDB.created_at.asc(), TenderMessageDB.id.asc()).all()

    return templates.TemplateResponse("employee_task_page.html", {
        "request": request,
        "current_employee": current_employee,
        "assignment": assignment,
        "tender": assignment.tender,
        "tasks": tasks,
        "team_members": team_members,
        "messages": messages,
        "task_comment_payloads": task_comment_payloads,
        "task_comment_counts": task_comment_counts,
        "now": datetime.utcnow()
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
        "employees": employees
    })


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

@router.post("/api/employee/messages")
async def send_message(
    request: Request,
    assignment_id: int = Form(...),
    message: str = Form(...),
    db: Session = Depends(get_db)

@router.get("/api/employee/messages/{assignment_id}")
async def get_messages(
    request: Request,
    assignment_id: int,
    since: Optional[str] = None,
    db: Session = Depends(get_db)

@router.get("/api/employee/messages/tender/{tender_id}")
async def get_tender_messages(
    request: Request,
    tender_id: str,
    since: Optional[str] = None,
    db: Session = Depends(get_db)

@router.post("/api/employee/assignments")
async def create_assignment(
    request: Request,
    tender_id: str = Form(...),
    employee_id: str = Form(...),
    role: str = Form(...),
    priority: str = Form("medium"),
    db: Session = Depends(get_db)

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

@router.post("/api/employee/assignments-with-tasks")
async def create_assignment_with_tasks(
    request: Request,
    tender_id: str = Form(...),
    employee_id: str = Form(...),
    role: str = Form(...),
    priority: str = Form("medium"),
    tasks: str = Form(...),
    db: Session = Depends(get_db)

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

# Certificate API endpoints

