/**
 * Expert Task Detail JavaScript
 * Handles task completion, file uploads, chat, progress tracking, and query raising
 */

// ==================== Page Load ====================
document.addEventListener('DOMContentLoaded', function() {
    loadTasks();
    loadMessages();
    setupFileUpload();
});

// ==================== Task Management ====================

async function loadTasks() {
    const container = document.getElementById('tasksContainer');
    container.innerHTML = '<p style="text-align: center; color: var(--text-muted);">Loading tasks...</p>';
    
    try {
        const response = await fetch(`/api/expert/projects/${channelType}/${channelId}/tasks`);
        if (!response.ok) throw new Error('Failed to load tasks');
        
        const data = await response.json();
        renderTasks(data.tasks || []);
        updateStats(data.tasks || []);
    } catch (error) {
        console.error('Error loading tasks:', error);
        container.innerHTML = '<p style="text-align: center; color: var(--error-color);">Failed to load tasks.</p>';
    }
}

function renderTasks(tasks) {
    const container = document.getElementById('tasksContainer');
    const taskCount = document.getElementById('taskCountBadge');
    
    if (!tasks || tasks.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"></path>
                </svg>
                <h4>No tasks assigned yet</h4>
                <p>Tasks will appear here when assigned to you</p>
            </div>
        `;
        taskCount.textContent = '(0)';
        return;
    }
    
    taskCount.textContent = `(${tasks.length})`;
    
    let html = '';
    tasks.forEach(task => {
        html += renderTaskCard(task);
    });
    
    container.innerHTML = html;
}

function renderTaskCard(task) {
    const isAssignedToMe = task.assignee_expert_id === currentExpertId;
    const canComplete = isAssignedToMe && task.status !== 'completed' && projectActive;
    const isCompleted = task.status === 'completed';
    
    return `
        <div class="task-card priority-${task.priority}" id="task-${task.id}">
            <div class="task-header">
                <div>
                    <h4 class="task-title">${escapeHtml(task.title)}</h4>
                </div>
                <div class="task-badges">
                    <span class="task-badge status-${task.status}">${formatStatus(task.status)}</span>
                    <span class="task-badge">${task.priority ? task.priority.charAt(0).toUpperCase() + task.priority.slice(1) : 'Medium'}</span>
                </div>
            </div>

            <div class="task-meta">
                ${task.deadline ? `<span>⏰ ${formatDate(task.deadline)}</span>` : ''}
                ${task.assignee_name ? `<span>👤 ${escapeHtml(task.assignee_name)}</span>` : '<span>👤 Unassigned</span>'}
            </div>

            ${task.description ? `<div class="task-description">${escapeHtml(task.description)}</div>` : ''}

            <div class="task-actions">
                ${canComplete ? `
                    <button class="btn-sm btn-success" onclick="completeTask('${task.id}')">✓ Mark Complete</button>
                ` : ''}
                ${!isCompleted && projectActive ? `
                    <button class="btn-sm btn-secondary" onclick="uploadFile('${task.id}')">📎 Upload Deliverable</button>
                    <button class="btn-sm btn-secondary" onclick="raiseQuery('${task.id}')">❓ Raise Query</button>
                ` : ''}
            </div>

            ${!isCompleted && isAssignedToMe ? renderProgressSection(task) : ''}
            ${task.files && task.files.length > 0 ? renderFilesSection(task) : ''}
        </div>
    `;
}

function renderProgressSection(task) {
    const progressCount = (task.progress_updates || []).length;
    
    return `
        <div class="progress-section">
            <div class="progress-header" onclick="toggleProgress('${task.id}')">
                <h5>
                    <span>📝</span>
                    <span>Work Progress Log</span>
                    <span style="color: var(--text-muted); font-weight: normal;">(${progressCount})</span>
                </h5>
                <span class="progress-toggle" id="progress-toggle-${task.id}">▼</span>
            </div>

            <div class="progress-content" id="progress-content-${task.id}">
                <!-- Progress Form -->
                <div class="progress-form">
                    <textarea
                        id="progress-input-${task.id}"
                        placeholder="Describe what you're working on or what you've accomplished..."
                        maxlength="2000"
                        oninput="updateCharCounter('${task.id}')"
                    ></textarea>
                    <div class="progress-form-footer">
                        <span class="char-counter" id="char-counter-${task.id}">0 / 2000</span>
                        <button class="btn-sm btn-primary" onclick="addProgressUpdate('${task.id}')">
                            Add Update
                        </button>
                    </div>
                </div>

                <!-- Progress List -->
                <div class="progress-list" id="progress-list-${task.id}">
                    ${task.progress_updates && task.progress_updates.length > 0 
                        ? task.progress_updates.map(p => renderProgressItem(p, task.id)).join('')
                        : renderEmptyProgress(task.id)
                    }
                </div>
            </div>
        </div>
    `;
}

function renderProgressItem(progress, taskId) {
    const isCurrentUser = progress.expert_id === currentExpertId || progress.user_id === currentUserId;
    const deleteButton = isCurrentUser ? `
        <div class="progress-actions">
            <button class="progress-delete-btn" onclick="deleteProgressUpdate('${taskId}', '${progress.id}')">
                Delete
            </button>
        </div>
    ` : '';
    
    return `
        <div class="progress-item" id="progress-${progress.id}">
            <div class="progress-item-header">
                <div class="progress-avatar">${progress.author_name ? progress.author_name[0].toUpperCase() : '?'}</div>
                <div class="progress-author-info">
                    <div class="progress-author-name">${escapeHtml(progress.author_name || 'Unknown')}</div>
                    <div class="progress-date">${formatDateTime(progress.created_at)}</div>
                </div>
            </div>
            <div class="progress-text">${escapeHtml(progress.update_text)}</div>
            ${deleteButton}
        </div>
    `;
}

function renderEmptyProgress(taskId) {
    return `
        <div class="progress-empty" id="progress-empty-${taskId}">
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
            </svg>
            <p>No progress updates yet. Start logging your work!</p>
        </div>
    `;
}

function renderFilesSection(task) {
    if (!task.files || task.files.length === 0) return '';
    
    return `
        <div class="files-section">
            <h5 style="font-size: 0.9rem; margin-bottom: 0.75rem;">Uploaded Deliverables</h5>
            <div class="files-list">
                ${task.files.map(file => `
                    <div class="file-item">
                        <div class="file-info">
                            <div class="file-icon">📄</div>
                            <div class="file-details">
                                <h6>${escapeHtml(file.filename)}</h6>
                                <p>${formatFileSize(file.file_size)} • ${formatDate(file.created_at)}</p>
                            </div>
                        </div>
                        <a href="/api/expert/projects/${channelType}/${channelId}/tasks/${task.id}/files/${file.id}/download" 
                           class="btn-sm btn-secondary" style="text-decoration: none;">Download</a>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
}

function updateStats(tasks) {
    const total = tasks.length;
    const completed = tasks.filter(t => t.status === 'completed').length;
    const pending = tasks.filter(t => t.status === 'pending').length;
    const inProgress = tasks.filter(t => t.status === 'in_progress').length;
    
    document.getElementById('totalTasksCount').textContent = total;
    document.getElementById('completedTasksCount').textContent = completed;
    document.getElementById('pendingTasksCount').textContent = pending;
    document.getElementById('inProgressTasksCount').textContent = inProgress;
}

// ==================== Task Actions ====================

async function completeTask(taskId) {
    if (!confirm('Mark this task as complete?')) {
        return;
    }
    
    try {
        const response = await fetch(`/api/expert/projects/${channelType}/${channelId}/tasks/${taskId}/status`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ status: 'completed' })
        });
        
        if (!response.ok) throw new Error('Failed to complete task');
        
        alert('Task marked as complete!');
        loadTasks();
    } catch (error) {
        console.error('Error completing task:', error);
        alert('Error: ' + error.message);
    }
}

// ==================== Progress Tracking ====================

function toggleProgress(taskId) {
    const content = document.getElementById(`progress-content-${taskId}`);
    const toggle = document.getElementById(`progress-toggle-${taskId}`);
    
    if (content && toggle) {
        content.classList.toggle('show');
        toggle.classList.toggle('expanded');
    }
}

function updateCharCounter(taskId) {
    const textarea = document.getElementById(`progress-input-${taskId}`);
    const counter = document.getElementById(`char-counter-${taskId}`);
    
    if (textarea && counter) {
        const length = textarea.value.length;
        const maxLength = 2000;
        
        counter.textContent = `${length} / ${maxLength}`;
        
        counter.classList.remove('warning', 'error');
        if (length > maxLength * 0.9) {
            counter.classList.add('error');
        } else if (length > maxLength * 0.7) {
            counter.classList.add('warning');
        }
    }
}

async function addProgressUpdate(taskId) {
    const textarea = document.getElementById(`progress-input-${taskId}`);
    const updateText = textarea.value.trim();
    
    if (!updateText) {
        alert('Please enter a progress update');
        return;
    }
    
    if (updateText.length > 2000) {
        alert('Progress update is too long (max 2000 characters)');
        return;
    }
    
    try {
        const response = await fetch(`/api/expert/projects/${channelType}/${channelId}/tasks/${taskId}/progress`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ update_text: updateText })
        });
        
        if (!response.ok) throw new Error('Failed to add progress update');
        
        const data = await response.json();
        
        // Clear the textarea
        textarea.value = '';
        updateCharCounter(taskId);
        
        // Update the UI
        const progressList = document.getElementById(`progress-list-${taskId}`);
        const emptyState = document.getElementById(`progress-empty-${taskId}`);
        
        // Remove empty state if it exists
        if (emptyState) {
            emptyState.remove();
        }
        
        // Add new progress item
        const progressItem = renderProgressItem(data.update, taskId);
        if (progressList) {
            progressList.insertAdjacentHTML('afterbegin', progressItem);
        }
        
        console.log('Progress update added successfully');
        
    } catch (error) {
        console.error('Error adding progress update:', error);
        alert('Error: ' + error.message);
    }
}

async function deleteProgressUpdate(taskId, progressId) {
    if (!confirm('Are you sure you want to delete this progress update?')) {
        return;
    }
    
    try {
        const response = await fetch(`/api/expert/projects/${channelType}/${channelId}/tasks/${taskId}/progress/${progressId}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) throw new Error('Failed to delete progress update');
        
        // Remove from DOM
        const progressItem = document.getElementById(`progress-${progressId}`);
        if (progressItem) {
            progressItem.remove();
        }
        
        // Check if list is empty and show empty state
        const progressList = document.getElementById(`progress-list-${taskId}`);
        if (progressList && progressList.children.length === 0) {
            progressList.innerHTML = renderEmptyProgress(taskId);
        }
        
        console.log('Progress update deleted successfully');
        
    } catch (error) {
        console.error('Error deleting progress update:', error);
        alert('Error: ' + error.message);
    }
}

// ==================== File Upload ====================

function uploadFile(taskId) {
    document.getElementById('uploadTaskId').value = taskId;
    document.getElementById('fileUploadModal').classList.add('active');
}

function closeFileUploadModal() {
    document.getElementById('fileUploadModal').classList.remove('active');
    document.getElementById('fileUploadForm').reset();
    document.getElementById('fileName').textContent = '';
}

function setupFileUpload() {
    const fileInput = document.getElementById('fileInput');
    const fileName = document.getElementById('fileName');
    
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const fileSize = (file.size / 1024).toFixed(2);
                fileName.textContent = `${file.name} (${fileSize} KB)`;
            } else {
                fileName.textContent = '';
            }
        });
        
        // Drag and drop
        const uploadArea = fileInput.closest('.file-upload-area');
        if (uploadArea) {
            uploadArea.addEventListener('dragover', function(e) {
                e.preventDefault();
                this.style.borderColor = 'var(--primary-color)';
                this.style.background = 'rgba(59, 130, 246, 0.1)';
            });
            
            uploadArea.addEventListener('dragleave', function(e) {
                e.preventDefault();
                this.style.borderColor = 'var(--border-color)';
                this.style.background = '';
            });
            
            uploadArea.addEventListener('drop', function(e) {
                e.preventDefault();
                this.style.borderColor = 'var(--border-color)';
                this.style.background = '';
                
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    fileInput.files = files;
                    const file = files[0];
                    const fileSize = (file.size / 1024).toFixed(2);
                    fileName.textContent = `${file.name} (${fileSize} KB)`;
                }
            });
        }
    }
}

async function uploadTaskFile(event) {
    event.preventDefault();
    const form = event.target;
    const formData = new FormData(form);
    const taskId = formData.get('task_id');
    
    try {
        const response = await fetch(`/api/expert/projects/${channelType}/${channelId}/tasks/${taskId}/files`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) throw new Error('Failed to upload file');
        
        alert('File uploaded successfully!');
        closeFileUploadModal();
        loadTasks();
    } catch (error) {
        console.error('Upload error:', error);
        alert('Error: ' + error.message);
    }
}

// ==================== Query/Concern Raising ====================

function raiseQuery(taskId) {
    document.getElementById('queryTaskId').value = taskId;
    document.getElementById('queryModal').classList.add('active');
}

function closeQueryModal() {
    document.getElementById('queryModal').classList.remove('active');
    document.getElementById('queryForm').reset();
}

async function submitQuery(event) {
    event.preventDefault();
    const form = event.target;
    const formData = new FormData(form);
    const taskId = formData.get('task_id');
    
    const payload = {
        query_type: formData.get('query_type'),
        title: formData.get('title'),
        description: formData.get('description'),
        priority: formData.get('priority')
    };
    
    try {
        const response = await fetch(`/api/expert/projects/${channelType}/${channelId}/tasks/${taskId}/queries`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        if (!response.ok) throw new Error('Failed to submit query');
        
        alert('Query submitted successfully! The project manager will be notified.');
        closeQueryModal();
    } catch (error) {
        console.error('Submit query error:', error);
        alert('Error: ' + error.message);
    }
}

// ==================== Chat Functions ====================

async function loadMessages() {
    const container = document.getElementById('chatMessages');
    container.innerHTML = '<p style="text-align: center; color: var(--text-muted);">Loading messages...</p>';
    
    try {
        const response = await fetch(`/api/expert/projects/${channelType}/${channelId}/messages`);
        if (!response.ok) throw new Error('Failed to load messages');
        
        const data = await response.json();
        renderMessages(data.messages || []);
    } catch (error) {
        console.error('Error loading messages:', error);
        container.innerHTML = '<p style="text-align: center; color: var(--error-color);">Failed to load chat.</p>';
    }
}

function renderMessages(messages) {
    const container = document.getElementById('chatMessages');
    
    if (!messages || messages.length === 0) {
        container.innerHTML = '<div class="empty-state"><p>No messages yet. Start the conversation!</p></div>';
        return;
    }
    
    let html = '';
    messages.forEach(message => {
        const isFromMe = (currentExpertId && message.sender_id === currentExpertId) || 
                        (currentUserId && message.sender_id === currentUserId);
        const messageClass = isFromMe ? 'from-expert' : 'from-manager';
        
        html += `
            <div class="chat-message ${messageClass}">
                <div class="chat-sender">
                    ${escapeHtml(message.sender || 'Team')}${isFromMe ? ' (You)' : ''}
                </div>
                <div class="chat-text">${escapeHtml(message.message)}</div>
                <div class="chat-time">${formatDateTime(message.created_at)}</div>
            </div>
        `;
    });
    
    container.innerHTML = html;
    container.scrollTop = container.scrollHeight;
}

async function sendMessage(event) {
    event.preventDefault();
    const input = document.getElementById('chatInput');
    const message = input.value.trim();
    
    if (!message) {
        return;
    }
    
    try {
        const response = await fetch(`/api/expert/projects/${channelType}/${channelId}/messages`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message })
        });
        
        if (!response.ok) throw new Error('Failed to send message');
        
        const data = await response.json();
        
        // Add message to chat immediately
        const chatMessages = document.getElementById('chatMessages');
        const messageDiv = document.createElement('div');
        messageDiv.className = 'chat-message from-expert';
        messageDiv.innerHTML = `
            <div class="chat-sender">You</div>
            <div class="chat-text">${escapeHtml(message)}</div>
            <div class="chat-time">Just now</div>
        `;
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        input.value = '';
    } catch (error) {
        console.error('Send message error:', error);
        alert('Error: ' + error.message);
    }
}

// ==================== Modal Management ====================

window.onclick = function(event) {
    const modals = document.querySelectorAll('.modal');
    modals.forEach(modal => {
        if (event.target === modal) {
            modal.classList.remove('active');
        }
    });
}

// ==================== Utility Functions ====================

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatStatus(status) {
    return status.replace('_', ' ').split(' ').map(word => 
        word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ');
}

function formatDate(dateString) {
    if (!dateString) return 'N/A';
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { 
        day: 'numeric', 
        month: 'short', 
        year: 'numeric' 
    });
}

function formatDateTime(dateString) {
    if (!dateString) return 'N/A';
    const date = new Date(dateString);
    return date.toLocaleString('en-US', { 
        day: 'numeric', 
        month: 'short', 
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

function formatFileSize(bytes) {
    if (!bytes) return '0 KB';
    const kb = bytes / 1024;
    if (kb < 1024) {
        return kb.toFixed(2) + ' KB';
    }
    return (kb / 1024).toFixed(2) + ' MB';
}

