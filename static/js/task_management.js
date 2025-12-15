/**
 * Task Management JavaScript
 * Handles all interactive functionality for the task workspace page
 */

// ==================== Modal Functions ====================

function openEmployeeModal() {
    document.getElementById('employeeModal').classList.add('active');
}

function closeEmployeeModal() {
    document.getElementById('employeeModal').classList.remove('active');
    document.getElementById('assignEmployeeForm').reset();
}

function openEditTaskModal() {
    document.getElementById('editTaskModal').classList.add('active');
}

function closeEditTaskModal() {
    document.getElementById('editTaskModal').classList.remove('active');
    document.getElementById('editTaskForm').reset();
}

function openSubtaskModal() {
    document.getElementById('subtaskModal').classList.add('active');
}

function closeSubtaskModal() {
    document.getElementById('subtaskModal').classList.remove('active');
    document.getElementById('subtaskForm').reset();
}

function openResolveConcernModal() {
    document.getElementById('resolveConcernModal').classList.add('active');
}

function closeResolveConcernModal() {
    document.getElementById('resolveConcernModal').classList.remove('active');
    document.getElementById('resolveConcernForm').reset();
}

// Close modals when clicking outside
window.onclick = function(event) {
    const modals = document.querySelectorAll('.modal');
    modals.forEach(modal => {
        if (event.target === modal) {
            modal.classList.remove('active');
        }
    });
}

// ==================== API Helper Functions ====================

async function apiCall(url, options = {}) {
    try {
        const response = await fetch(url, {
            ...options,
            headers: {
                ...options.headers,
            },
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'An error occurred' }));
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error('API call error:', error);
        alert(`Error: ${error.message}`);
        throw error;
    }
}

function showSuccess(message) {
    // You can replace this with a nicer toast notification
    alert(message);
}

// ==================== Task Management Functions ====================

async function createTask(event) {
    event.preventDefault();
    const form = event.target;
    const formData = new FormData(form);

    // Add tender_id from the global variable
    formData.append('tender_id', tenderId);

    try {
        const response = await fetch('/api/tasks/create', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'Failed to create task' }));
            throw new Error(errorData.detail || 'Failed to create task');
        }

        const data = await response.json();
        showSuccess('Task created successfully!');

        // Reload the page to show the new task
        window.location.reload();
    } catch (error) {
        console.error('Error creating task:', error);
        alert(`Error: ${error.message}`);
    }
}

function editTask(taskId) {
    const taskData = tasksData[taskId];
    if (!taskData) {
        alert('Task data not found');
        return;
    }

    // Populate the edit form
    document.getElementById('editTaskId').value = taskId;
    document.getElementById('editTaskTitle').value = taskData.title;
    document.getElementById('editTaskDescription').value = taskData.description;
    document.getElementById('editTaskPriority').value = taskData.priority;
    document.getElementById('editTaskStatus').value = taskData.status;
    document.getElementById('editTaskDeadline').value = taskData.deadline;

    openEditTaskModal();
}

async function updateTask(event) {
    event.preventDefault();
    const form = event.target;
    const formData = new FormData(form);
    const taskId = formData.get('task_id');

    try {
        const response = await fetch(`/api/tasks/${taskId}`, {
            method: 'PUT',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'Failed to update task' }));
            throw new Error(errorData.detail || 'Failed to update task');
        }

        showSuccess('Task updated successfully!');
        closeEditTaskModal();

        // Reload the page to show the updated task
        window.location.reload();
    } catch (error) {
        console.error('Error updating task:', error);
        alert(`Error: ${error.message}`);
    }
}

async function deleteTask(taskId) {
    if (!confirm('Are you sure you want to delete this task? This will also delete all subtasks.')) {
        return;
    }

    try {
        const response = await fetch(`/api/tasks/${taskId}`, {
            method: 'DELETE'
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'Failed to delete task' }));
            throw new Error(errorData.detail || 'Failed to delete task');
        }

        showSuccess('Task deleted successfully!');

        // Remove the task card from DOM
        const taskCard = document.getElementById(`task-${taskId}`);
        if (taskCard) {
            taskCard.remove();
        }

        // Reload to update counts
        setTimeout(() => window.location.reload(), 500);
    } catch (error) {
        console.error('Error deleting task:', error);
        alert(`Error: ${error.message}`);
    }
}

async function updateTaskStatus(taskId, status) {
    const formData = new FormData();
    formData.append('status', status);

    try {
        const response = await fetch(`/api/tasks/${taskId}`, {
            method: 'PUT',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'Failed to update task status' }));
            throw new Error(errorData.detail || 'Failed to update task status');
        }

        showSuccess(`Task marked as ${status}!`);

        // Reload the page to show the updated task
        window.location.reload();
    } catch (error) {
        console.error('Error updating task status:', error);
        alert(`Error: ${error.message}`);
    }
}

// ==================== Subtask Functions ====================

function addSubtask(parentTaskId) {
    document.getElementById('subtaskParentId').value = parentTaskId;
    openSubtaskModal();
}

async function createSubtask(event) {
    event.preventDefault();
    const form = event.target;
    const formData = new FormData(form);

    try {
        const response = await fetch('/api/tasks/subtask', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'Failed to create subtask' }));
            throw new Error(errorData.detail || 'Failed to create subtask');
        }

        const data = await response.json();
        showSuccess('Subtask created successfully!');
        closeSubtaskModal();

        // Reload the page to show the new subtask
        window.location.reload();
    } catch (error) {
        console.error('Error creating subtask:', error);
        alert(`Error: ${error.message}`);
    }
}

// ==================== Employee Assignment Functions ====================

async function assignEmployees(event) {
    event.preventDefault();
    const form = event.target;
    const formData = new FormData(form);

    const selectedEmployeeIds = formData.getAll('employee_ids');
    const role = formData.get('role');

    if (selectedEmployeeIds.length === 0) {
        alert('Please select at least one employee');
        return;
    }

    if (!role.trim()) {
        alert('Please enter a role');
        return;
    }

    try {
        // Create assignments for each selected employee
        const promises = selectedEmployeeIds.map(employeeId => {
            const assignmentData = new FormData();
            assignmentData.append('tender_id', tenderId);
            assignmentData.append('employee_id', employeeId);
            assignmentData.append('role', role);
            assignmentData.append('priority', 'medium');

            return fetch('/api/tender/assign', {
                method: 'POST',
                body: assignmentData
            });
        });

        const responses = await Promise.all(promises);

        // Check if all assignments succeeded
        const failedAssignments = responses.filter(r => !r.ok);
        if (failedAssignments.length > 0) {
            throw new Error(`Failed to assign ${failedAssignments.length} employee(s)`);
        }

        showSuccess(`Successfully assigned ${selectedEmployeeIds.length} employee(s)!`);
        closeEmployeeModal();

        // Reload the page to show the new team members
        window.location.reload();
    } catch (error) {
        console.error('Error assigning employees:', error);
        alert(`Error: ${error.message}`);
    }
}

// ==================== Concern Functions ====================

function resolveConcern(concernId) {
    document.getElementById('resolveConcernId').value = concernId;
    openResolveConcernModal();
}

async function submitResolveConcern(event) {
    event.preventDefault();
    const form = event.target;
    const formData = new FormData(form);
    const concernId = formData.get('concern_id');
    const resolutionNotes = formData.get('resolution_notes');

    if (!resolutionNotes.trim()) {
        alert('Please enter resolution notes');
        return;
    }

    try {
        const response = await fetch(`/api/concerns/${concernId}/resolve`, {
            method: 'PUT',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'Failed to resolve concern' }));
            throw new Error(errorData.detail || 'Failed to resolve concern');
        }

        showSuccess('Concern marked as resolved!');
        closeResolveConcernModal();

        // Reload the page to show the resolved concern
        window.location.reload();
    } catch (error) {
        console.error('Error resolving concern:', error);
        alert(`Error: ${error.message}`);
    }
}

// ==================== Chat Functions ====================

// Track if a message is currently being sent to prevent double-submit
let isSendingMessage = false;

async function sendMessage(event) {
    event.preventDefault();

    // Prevent double-submit
    if (isSendingMessage) {
        console.log('Message already being sent, ignoring');
        return;
    }

    const form = event.target;
    const messageInput = document.getElementById('chatInput');
    const submitBtn = form.querySelector('button[type="submit"]');
    const message = messageInput.value.trim();

    if (!message) {
        return;
    }

    // Store original button state and disable
    const originalBtnText = submitBtn ? submitBtn.textContent : 'Send';
    if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.textContent = 'Sending...';
    }

    // Mark as sending
    isSendingMessage = true;

    try {
        // Add the message to the chat immediately (optimistic UI update) BEFORE API call
        const chatMessages = document.getElementById('chatMessages');

        // Remove empty state if it exists
        const emptyState = chatMessages.querySelector('.empty-state');
        if (emptyState) {
            emptyState.remove();
        }

        const tempId = `temp-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        const messageHTML = `
            <div class="chat-message manager" data-temp-id="${tempId}" data-message-text="${escapeHtml(message)}">
                <div class="chat-sender">You</div>
                <div class="chat-text">${escapeHtml(message)}</div>
                <div class="chat-time">Just now</div>
            </div>
        `;
        chatMessages.insertAdjacentHTML('beforeend', messageHTML);

        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;

        // Clear the input immediately for better UX
        messageInput.value = '';

        // We need to send to an assignment - get the first assignment for this tender
        // The backend will handle this logic, but we need an endpoint to post messages
        const formData = new FormData();
        formData.append('tender_id', tenderId);
        formData.append('message', message);

        const response = await fetch('/api/tender/message', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            // Remove the optimistic message on error
            const tempMsg = chatMessages.querySelector(`[data-temp-id="${tempId}"]`);
            if (tempMsg) {
                tempMsg.remove();
            }
            messageInput.value = message; // Restore the message

            const errorData = await response.json().catch(() => ({ detail: 'Failed to send message' }));
            if (response.status === 400) {
                alert('Please add team members to the tender before using chat');
                return;
            }
            throw new Error(errorData.detail || 'Failed to send message');
        }

        // Message sent successfully - the WebSocket will handle updating it

    } catch (error) {
        console.error('Error sending message:', error);
        alert(`Error: ${error.message}`);
    } finally {
        // Reset sending flag
        isSendingMessage = false;

        // Always reset button state
        if (submitBtn) {
            submitBtn.textContent = originalBtnText;
            submitBtn.disabled = false;
        }
    }
}

// Helper function to escape HTML and prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ==================== Initialization ====================

document.addEventListener('DOMContentLoaded', function() {
    // Auto-scroll chat to bottom on page load
    const chatMessages = document.getElementById('chatMessages');
    if (chatMessages) {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Add click handlers to employee items in the modal to toggle selection
    document.querySelectorAll('.employee-item').forEach(item => {
        item.addEventListener('click', function(e) {
            if (e.target.type !== 'checkbox') {
                const checkbox = this.querySelector('input[type="checkbox"]');
                if (checkbox) {
                    checkbox.checked = !checkbox.checked;
                    this.classList.toggle('selected', checkbox.checked);
                }
            } else {
                this.classList.toggle('selected', e.target.checked);
            }
        });
    });

    // Initialize selected state for checkboxes
    document.querySelectorAll('.employee-item input[type="checkbox"]').forEach(checkbox => {
        if (checkbox.checked) {
            checkbox.closest('.employee-item').classList.add('selected');
        }
    });

    // Initialize WebSocket connection
    if (typeof tenderId !== 'undefined' && typeof currentManagerId !== 'undefined' && typeof currentManagerName !== 'undefined') {
        initWebSocket();
    } else {
        console.error('Missing required variables for WebSocket connection');
    }
});

// ==================== WebSocket Connection ====================

let ws = null;
let reconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 5;
const RECONNECT_DELAY = 3000; // 3 seconds
let pingInterval = null;

/**
 * Initialize WebSocket connection for real-time updates
 */
function initWebSocket() {
    // Determine WebSocket protocol (ws or wss based on page protocol)
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/tender/${tenderId}`;

    console.log(`Connecting to WebSocket: ${wsUrl}`);

    try {
        ws = new WebSocket(wsUrl);

        ws.onopen = function(event) {
            console.log('WebSocket connected');
            reconnectAttempts = 0;

            // Send authentication message
            const authMessage = {
                type: 'auth',
                user_type: 'manager',
                user_id: currentManagerId,
                user_name: currentManagerName
            };
            ws.send(JSON.stringify(authMessage));

            // Start ping interval for keep-alive
            startPingInterval();
        };

        ws.onmessage = function(event) {
            try {
                const message = JSON.parse(event.data);
                console.log('WebSocket message received:', message);

                handleWebSocketMessage(message);
            } catch (error) {
                console.error('Error parsing WebSocket message:', error);
            }
        };

        ws.onerror = function(error) {
            console.error('WebSocket error:', error);
        };

        ws.onclose = function(event) {
            console.log('WebSocket disconnected');
            stopPingInterval();

            // Attempt to reconnect if not closed intentionally
            if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
                reconnectAttempts++;
                console.log(`Reconnecting... Attempt ${reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS}`);
                setTimeout(initWebSocket, RECONNECT_DELAY);
            } else {
                console.error('Max reconnection attempts reached');
                showConnectionError();
            }
        };

    } catch (error) {
        console.error('Error initializing WebSocket:', error);
    }
}

/**
 * Handle incoming WebSocket messages
 */
function handleWebSocketMessage(message) {
    switch (message.type) {
        case 'connected':
            console.log('WebSocket authenticated successfully');
            hideConnectionError();
            break;

        case 'chat_message':
            handleIncomingChatMessage(message.data);
            break;

        case 'task_created':
        case 'task_updated':
        case 'task_deleted':
            handleTaskEvent(message.type, message.data);
            break;

        case 'user_connected':
            console.log(`${message.user_name} (${message.user_type}) connected`);
            break;

        case 'pong':
            // Keep-alive response
            break;

        case 'error':
            console.error('WebSocket error:', message.message);
            break;

        default:
            console.warn('Unknown message type:', message.type);
    }
}

/**
 * Handle incoming chat message
 */
function handleIncomingChatMessage(data) {
    const chatMessages = document.getElementById('chatMessages');
    if (!chatMessages) return;

    // Remove empty state if it exists
    const emptyState = chatMessages.querySelector('.empty-state');
    if (emptyState) {
        emptyState.remove();
    }

    // Check if this is our own message (avoid duplicates from optimistic update)
    const isOwnMessage = (data.sender_type === 'manager');

    // If it's our own message, look for the temporary optimistic message and replace it
    // Use .textContent comparison instead of attribute (more reliable)
    if (isOwnMessage) {
        const tempMessages = chatMessages.querySelectorAll('[data-temp-id]');
        for (const tempMsg of tempMessages) {
            // Compare using the actual text content (handles HTML escaping properly)
            const tempTextElement = tempMsg.querySelector('.chat-text');
            const tempText = tempTextElement ? tempTextElement.textContent.trim() : '';

            if (tempText === data.message.trim()) {
                // Found the optimistic message - update it with the real timestamp
                const timeDiv = tempMsg.querySelector('.chat-time');
                if (timeDiv) {
                    timeDiv.textContent = data.formatted_time;
                }
                // Remove the temp attributes
                tempMsg.removeAttribute('data-temp-id');
                tempMsg.removeAttribute('data-message-text');
                console.log('Updated optimistic message with real timestamp');
                return; // Don't add a new message
            }
        }
    }

    // Check for duplicate: look for recent message with same text (within last 5 messages for better coverage)
    const existingMessages = Array.from(chatMessages.querySelectorAll('.chat-message')).slice(-5);
    const isDuplicate = existingMessages.some(msg => {
        const msgTextElement = msg.querySelector('.chat-text');
        const msgText = msgTextElement ? msgTextElement.textContent.trim() : '';
        const msgSenderElement = msg.querySelector('.chat-sender');
        const msgSender = msgSenderElement ? msgSenderElement.textContent.trim() : '';

        // Check if it's the same message (same text and sender)
        if (isOwnMessage) {
            return msgText === data.message.trim() && (msgSender === 'You' || msgSender.includes('(You)'));
        } else {
            return msgText === data.message.trim() && msgSender.includes(data.sender_name);
        }
    });

    if (isDuplicate) {
        console.log('Duplicate message detected, skipping');
        return;
    }

    // Create message element
    const messageDiv = document.createElement('div');
    const messageClass = isOwnMessage ? 'chat-message manager' : 'chat-message employee';

    messageDiv.className = messageClass;
    messageDiv.innerHTML = `
        <div class="chat-sender">${isOwnMessage ? 'You' : escapeHtml(data.sender_name)}</div>
        <div class="chat-text">${escapeHtml(data.message)}</div>
        <div class="chat-time">${data.formatted_time}</div>
    `;

    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

/**
 * Handle task-related events
 */
function handleTaskEvent(eventType, data) {
    console.log(`Task event: ${eventType}`, data);

    // For now, just reload the page to show updates
    // TODO: Implement live DOM updates without page reload
    if (eventType === 'task_created' || eventType === 'task_updated' || eventType === 'task_deleted') {
        // Show notification (optional)
        console.log('Task changed, consider refreshing the page');

        // You can add a notification banner here
        // For now, we'll skip auto-reload to avoid disrupting user work
    }
}

/**
 * Start ping interval for keep-alive
 */
function startPingInterval() {
    pingInterval = setInterval(() => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'ping' }));
        }
    }, 30000); // Ping every 30 seconds
}

/**
 * Stop ping interval
 */
function stopPingInterval() {
    if (pingInterval) {
        clearInterval(pingInterval);
        pingInterval = null;
    }
}

/**
 * Show connection error notification
 */
function showConnectionError() {
    const chatContainer = document.querySelector('.workspace-chat');
    if (!chatContainer) return;

    let errorBanner = document.getElementById('ws-error-banner');
    if (!errorBanner) {
        errorBanner = document.createElement('div');
        errorBanner.id = 'ws-error-banner';
        errorBanner.style.cssText = 'background: #fee; color: #c33; padding: 0.5rem; border-radius: 4px; margin-bottom: 0.5rem; font-size: 0.9rem; text-align: center;';
        errorBanner.textContent = 'Connection lost. Chat updates are paused.';
        chatContainer.insertBefore(errorBanner, chatContainer.firstChild);
    }
}

/**
 * Hide connection error notification
 */
function hideConnectionError() {
    const errorBanner = document.getElementById('ws-error-banner');
    if (errorBanner) {
        errorBanner.remove();
    }
}

/**
 * Close WebSocket connection
 */
function closeWebSocket() {
    if (ws) {
        stopPingInterval();
        ws.close();
        ws = null;
    }
}

// Clean up WebSocket on page unload
window.addEventListener('beforeunload', function() {
    closeWebSocket();
});
