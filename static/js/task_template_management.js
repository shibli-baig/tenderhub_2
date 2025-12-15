/**
 * Task Template Management
 * Handles CRUD operations for automated task templates
 */

let currentStage = 1;
let templateCache = {};

// ==================== Modal Functions ====================

function openTaskTemplateModal() {
    document.getElementById('taskTemplateModal').classList.add('is-visible');
    loadAllTemplates();
}

function closeTaskTemplateModal() {
    document.getElementById('taskTemplateModal').classList.remove('is-visible');
}

function switchStageTab(stageNumber) {
    // Update tab active state
    document.querySelectorAll('.stage-tab').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelector(`[data-stage="${stageNumber}"]`).classList.add('active');

    // Update content active state
    document.querySelectorAll('.stage-content').forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById(`stageContent${stageNumber}`).classList.add('active');

    currentStage = stageNumber;
}

// ==================== Template Loading ====================

async function loadAllTemplates() {
    try {
        const response = await fetch('/api/task-templates');
        const data = await response.json();

        if (data.success) {
            // Clear cache and lists
            templateCache = {};
            for (let i = 1; i <= 6; i++) {
                document.getElementById(`taskList${i}`).innerHTML = '';
            }

            // Organize by stage
            data.templates.forEach(template => {
                if (!templateCache[template.stage_number]) {
                    templateCache[template.stage_number] = [];
                }
                templateCache[template.stage_number].push(template);
            });

            // Render each stage
            for (let i = 1; i <= 6; i++) {
                renderStageTemplates(i);
            }
        }
    } catch (error) {
        console.error('Error loading templates:', error);
        alert('Failed to load task templates');
    }
}

function renderStageTemplates(stageNumber) {
    const container = document.getElementById(`taskList${stageNumber}`);
    const templates = templateCache[stageNumber] || [];

    if (templates.length === 0) {
        container.innerHTML = '<p style="color: var(--text-muted); text-align: center; padding: 2rem;">No tasks defined for this stage. Click "+ Add Task" to create one.</p>';
        return;
    }

    container.innerHTML = templates.map(template => `
        <div class="task-template-card" data-template-id="${template.id}">
            <div class="task-template-header">
                <div style="flex: 1;">
                    <h4 style="margin: 0 0 0.5rem 0;">${template.task_title}</h4>
                    <p style="margin: 0; color: var(--text-muted); font-size: 0.9rem;">${template.task_description || 'No description'}</p>
                </div>
                <div class="task-template-actions">
                    <button class="btn btn-sm btn-secondary" onclick="editTaskTemplate(${template.id}, ${stageNumber})" title="Edit">✏️</button>
                    <button class="btn btn-sm btn-danger" onclick="deleteTaskTemplate(${template.id}, ${stageNumber})" title="Delete">🗑️</button>
                </div>
            </div>
            <div style="display: flex; gap: 1rem; margin-top: 0.75rem; font-size: 0.85rem; color: var(--text-muted);">
                <span><strong>Priority:</strong> ${template.priority}</span>
                <span><strong>Deadline:</strong> ${template.deadline_days} days</span>
                ${template.estimated_hours ? `<span><strong>Est. Hours:</strong> ${template.estimated_hours}h</span>` : ''}
            </div>

            ${template.subtasks && template.subtasks.length > 0 ? `
                <div class="subtask-list">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                        <strong style="font-size: 0.9rem;">Subtasks (${template.subtasks.length})</strong>
                        <button class="btn btn-sm btn-outline" onclick="addSubtaskTemplate(${template.id}, ${stageNumber})">+ Add Subtask</button>
                    </div>
                    ${template.subtasks.map(subtask => `
                        <div class="subtask-card">
                            <div style="display: flex; justify-content: space-between; align-items: start;">
                                <div style="flex: 1;">
                                    <strong>${subtask.task_title}</strong>
                                    <p style="margin: 0.25rem 0 0 0; font-size: 0.85rem; color: var(--text-muted);">${subtask.task_description || ''}</p>
                                    <div style="margin-top: 0.5rem; font-size: 0.8rem; color: var(--text-muted);">
                                        Priority: ${subtask.priority} | Deadline: ${subtask.deadline_days} days
                                    </div>
                                </div>
                                <div style="display: flex; gap: 0.25rem;">
                                    <button class="btn btn-sm btn-secondary" onclick="editSubtaskTemplate(${subtask.id}, ${template.id}, ${stageNumber})" title="Edit">✏️</button>
                                    <button class="btn btn-sm btn-danger" onclick="deleteSubtaskTemplate(${subtask.id}, ${template.id}, ${stageNumber})" title="Delete">🗑️</button>
                                </div>
                            </div>
                        </div>
                    `).join('')}
                </div>
            ` : `
                <div style="margin-top: 1rem;">
                    <button class="btn btn-sm btn-outline" onclick="addSubtaskTemplate(${template.id}, ${stageNumber})">+ Add Subtask</button>
                </div>
            `}
        </div>
    `).join('');
}

// ==================== CRUD Operations ====================

async function addTaskTemplate(stageNumber) {
    const title = prompt('Task Title:');
    if (!title) return;

    const description = prompt('Task Description (optional):') || '';
    const priority = prompt('Priority (low/medium/high):', 'medium');
    const deadlineDays = parseInt(prompt('Days until deadline:', '7'));
    const estimatedHours = prompt('Estimated hours (optional):');

    if (isNaN(deadlineDays)) {
        alert('Invalid deadline days');
        return;
    }

    try {
        const response = await fetch('/api/task-templates', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                stage_number: stageNumber,
                task_title: title,
                task_description: description,
                priority: priority,
                deadline_days: deadlineDays,
                estimated_hours: estimatedHours ? parseFloat(estimatedHours) : null
            })
        });

        const data = await response.json();

        if (data.success) {
            await loadAllTemplates();
            switchStageTab(stageNumber);
        } else {
            alert('Failed to create task template');
        }
    } catch (error) {
        console.error('Error creating template:', error);
        alert('Failed to create task template');
    }
}

async function editTaskTemplate(templateId, stageNumber) {
    const templates = templateCache[stageNumber] || [];
    const template = templates.find(t => t.id === templateId);
    if (!template) return;

    const title = prompt('Task Title:', template.task_title);
    if (!title) return;

    const description = prompt('Task Description:', template.task_description || '');
    const priority = prompt('Priority (low/medium/high):', template.priority);
    const deadlineDays = parseInt(prompt('Days until deadline:', template.deadline_days));
    const estimatedHours = prompt('Estimated hours (optional):', template.estimated_hours || '');

    if (isNaN(deadlineDays)) {
        alert('Invalid deadline days');
        return;
    }

    try {
        const response = await fetch(`/api/task-templates/${templateId}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                task_title: title,
                task_description: description,
                priority: priority,
                deadline_days: deadlineDays,
                estimated_hours: estimatedHours ? parseFloat(estimatedHours) : null
            })
        });

        const data = await response.json();

        if (data.success) {
            await loadAllTemplates();
            switchStageTab(stageNumber);
        } else {
            alert('Failed to update task template');
        }
    } catch (error) {
        console.error('Error updating template:', error);
        alert('Failed to update task template');
    }
}

async function deleteTaskTemplate(templateId, stageNumber) {
    if (!confirm('Delete this task template? This will also delete all subtasks.')) return;

    try {
        const response = await fetch(`/api/task-templates/${templateId}`, {
            method: 'DELETE'
        });

        const data = await response.json();

        if (data.success) {
            await loadAllTemplates();
            switchStageTab(stageNumber);
        } else {
            alert('Failed to delete task template');
        }
    } catch (error) {
        console.error('Error deleting template:', error);
        alert('Failed to delete task template');
    }
}

// ==================== Subtask Operations ====================

async function addSubtaskTemplate(parentId, stageNumber) {
    const title = prompt('Subtask Title:');
    if (!title) return;

    const description = prompt('Subtask Description (optional):') || '';
    const priority = prompt('Priority (low/medium/high):', 'medium');
    const deadlineDays = parseInt(prompt('Days until deadline:', '7'));
    const estimatedHours = prompt('Estimated hours (optional):');

    if (isNaN(deadlineDays)) {
        alert('Invalid deadline days');
        return;
    }

    try {
        const response = await fetch('/api/task-templates', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                stage_number: stageNumber,
                task_title: title,
                task_description: description,
                priority: priority,
                deadline_days: deadlineDays,
                estimated_hours: estimatedHours ? parseFloat(estimatedHours) : null,
                parent_template_id: parentId
            })
        });

        const data = await response.json();

        if (data.success) {
            await loadAllTemplates();
            switchStageTab(stageNumber);
        } else {
            alert('Failed to create subtask template');
        }
    } catch (error) {
        console.error('Error creating subtask:', error);
        alert('Failed to create subtask template');
    }
}

async function editSubtaskTemplate(subtaskId, parentId, stageNumber) {
    const templates = templateCache[stageNumber] || [];
    const parentTemplate = templates.find(t => t.id === parentId);
    if (!parentTemplate) return;

    const subtask = parentTemplate.subtasks.find(st => st.id === subtaskId);
    if (!subtask) return;

    const title = prompt('Subtask Title:', subtask.task_title);
    if (!title) return;

    const description = prompt('Subtask Description:', subtask.task_description || '');
    const priority = prompt('Priority (low/medium/high):', subtask.priority);
    const deadlineDays = parseInt(prompt('Days until deadline:', subtask.deadline_days));
    const estimatedHours = prompt('Estimated hours (optional):', subtask.estimated_hours || '');

    if (isNaN(deadlineDays)) {
        alert('Invalid deadline days');
        return;
    }

    try {
        const response = await fetch(`/api/task-templates/${subtaskId}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                task_title: title,
                task_description: description,
                priority: priority,
                deadline_days: deadlineDays,
                estimated_hours: estimatedHours ? parseFloat(estimatedHours) : null
            })
        });

        const data = await response.json();

        if (data.success) {
            await loadAllTemplates();
            switchStageTab(stageNumber);
        } else {
            alert('Failed to update subtask template');
        }
    } catch (error) {
        console.error('Error updating subtask:', error);
        alert('Failed to update subtask template');
    }
}

async function deleteSubtaskTemplate(subtaskId, parentId, stageNumber) {
    if (!confirm('Delete this subtask template?')) return;

    try {
        const response = await fetch(`/api/task-templates/${subtaskId}`, {
            method: 'DELETE'
        });

        const data = await response.json();

        if (data.success) {
            await loadAllTemplates();
            switchStageTab(stageNumber);
        } else {
            alert('Failed to delete subtask template');
        }
    } catch (error) {
        console.error('Error deleting subtask:', error);
        alert('Failed to delete subtask template');
    }
}
