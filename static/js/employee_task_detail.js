/**
 * Employee Task Detail JavaScript
 * Handles task completion, file uploads, chat, and concern raising
 */

// ==================== Modal Functions ====================

function closeFileUploadModal() {
    document.getElementById('fileUploadModal').classList.remove('active');
    document.getElementById('fileUploadForm').reset();
    document.getElementById('fileName').textContent = '';
}

function closeConcernModal() {
    document.getElementById('concernModal').classList.remove('active');
    document.getElementById('concernForm').reset();
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

// ==================== File Upload ====================

function uploadFile(taskId) {
    document.getElementById('uploadTaskId').value = taskId;
    document.getElementById('fileUploadModal').classList.add('active');
}

// Handle file selection display
document.addEventListener('DOMContentLoaded', function() {
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
});

async function uploadTaskFile(event) {
    event.preventDefault();
    const form = event.target;
    const formData = new FormData(form);
    const taskId = formData.get('task_id');

    try {
        const response = await fetch(`/api/tasks/${taskId}/files`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'Failed to upload file' }));
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        alert('File uploaded successfully!');
        closeFileUploadModal();
        location.reload(); // Reload to show the new file
    } catch (error) {
        console.error('Upload error:', error);
        alert(`Error: ${error.message}`);
    }
}

// ==================== Task Completion ====================

async function completeTask(taskId) {
    if (!confirm('Mark this task as complete?')) {
        return;
    }

    try {
        const response = await fetch(`/api/employee/tasks/${taskId}/complete`, {
            method: 'POST'
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'Failed to complete task' }));
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }

        alert('Task marked as complete!');
        location.reload();
    } catch (error) {
        console.error('Complete task error:', error);
        alert(`Error: ${error.message}`);
    }
}

// ==================== Concern Raising ====================

function raiseConcern(taskId) {
    document.getElementById('concernTaskId').value = taskId;
    document.getElementById('concernModal').classList.add('active');
}

async function submitConcern(event) {
    event.preventDefault();
    const form = event.target;
    const formData = new FormData(form);
    const taskId = formData.get('task_id');

    try {
        const response = await fetch(`/api/tasks/${taskId}/concerns`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'Failed to submit concern' }));
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }

        alert('Query submitted successfully! Your manager will be notified.');
        closeConcernModal();
        location.reload();
    } catch (error) {
        console.error('Submit concern error:', error);
        alert(`Error: ${error.message}`);
    }
}

// ==================== Chat Functions ====================

async function sendMessage(event) {
    event.preventDefault();
    const input = document.getElementById('chatInput');
    const submitBtn = event.target.querySelector('button[type="submit"]');
    const message = input.value.trim();
    
    if (!message) {
        return;
    }

    // Store original button state
    const originalBtnText = submitBtn ? submitBtn.textContent : 'Send';
    
    try {
        // Add message to chat immediately (optimistic update) BEFORE API call
        const chatMessages = document.getElementById('chatMessages');
        const tempId = `temp-${Date.now()}`;
        const messageDiv = document.createElement('div');
        messageDiv.className = 'chat-message employee';
        messageDiv.setAttribute('data-temp-id', tempId);
        messageDiv.setAttribute('data-message-text', message);
        messageDiv.innerHTML = `
            <div class="chat-sender">You</div>
            <div class="chat-text">${escapeHtml(message)}</div>
            <div class="chat-time">Just now</div>
        `;
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        // Clear input immediately for better UX
        input.value = '';

        const formData = new FormData();
        formData.append('assignment_id', assignmentId);
        formData.append('message', message);

        const response = await fetch('/api/employee/messages', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            // Remove the optimistic message on error
            messageDiv.remove();
            input.value = message; // Restore the message
            
            const errorData = await response.json().catch(() => ({ detail: 'Failed to send message' }));
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }

        // Message sent successfully - the WebSocket will handle updating it
        
    } catch (error) {
        console.error('Send message error:', error);
        alert(`Error: ${error.message}`);
    } finally {
        // Always reset button state
        if (submitBtn) {
            submitBtn.textContent = originalBtnText;
            submitBtn.disabled = false;
        }
    }
}

// Auto-scroll chat to bottom on load
document.addEventListener('DOMContentLoaded', function() {
    const chatMessages = document.getElementById('chatMessages');
    if (chatMessages) {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
});

// ==================== Progress Tracking Functions ====================

/**
 * Toggle the visibility of the progress section for a task
 */
function toggleProgress(taskId) {
    const content = document.getElementById(`progress-content-${taskId}`);
    const toggle = document.getElementById(`progress-toggle-${taskId}`);

    if (content && toggle) {
        content.classList.toggle('show');
        toggle.classList.toggle('expanded');
    }
}

/**
 * Update the character counter for the progress input
 */
function updateCharCounter(taskId) {
    const textarea = document.getElementById(`progress-input-${taskId}`);
    const counter = document.getElementById(`char-counter-${taskId}`);

    if (textarea && counter) {
        const length = textarea.value.length;
        const maxLength = 2000;

        counter.textContent = `${length} / ${maxLength}`;

        // Update counter color based on character count
        counter.classList.remove('warning', 'error');
        if (length > maxLength * 0.9) {
            counter.classList.add('error');
        } else if (length > maxLength * 0.7) {
            counter.classList.add('warning');
        }
    }
}

/**
 * Add a new progress update to a task
 */
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
        const formData = new FormData();
        formData.append('update_text', updateText);

        const response = await fetch(`/api/tasks/${taskId}/progress`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'Failed to add progress update' }));
            throw new Error(errorData.detail || 'Failed to add progress update');
        }

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

        // Create new progress item HTML
        const progressItem = createProgressItemHTML(data.update);

        // Insert at the top of the list (newest first)
        if (progressList) {
            progressList.insertAdjacentHTML('afterbegin', progressItem);
        }

        // Update the count in the header
        const headerCount = document.querySelector(`#progress-content-${taskId}`).previousElementSibling.querySelector('span[style*="text-muted"]');
        if (headerCount) {
            const currentCount = parseInt(headerCount.textContent.replace(/[()]/g, '')) || 0;
            headerCount.textContent = `(${currentCount + 1})`;
        }

        // Show success message (optional)
        console.log('Progress update added successfully');

    } catch (error) {
        console.error('Error adding progress update:', error);
        alert(`Error: ${error.message}`);
    }
}

/**
 * Create HTML for a progress item
 */
function createProgressItemHTML(progress) {
    const deleteButton = progress.is_current_user ? `
        <div class="progress-actions">
            <button class="progress-delete-btn" onclick="deleteProgressUpdate(${progress.task_id}, ${progress.id})">
                Delete
            </button>
        </div>
    ` : '';

    return `
        <div class="progress-item" id="progress-${progress.id}">
            <div class="progress-item-header">
                <div class="progress-avatar">${progress.employee_avatar}</div>
                <div class="progress-author-info">
                    <div class="progress-author-name">${progress.employee_name}</div>
                    <div class="progress-date">${progress.formatted_date}</div>
                </div>
            </div>
            <div class="progress-text">${escapeHtml(progress.update_text)}</div>
            ${deleteButton}
        </div>
    `;
}

/**
 * Delete a progress update
 */
async function deleteProgressUpdate(taskId, progressId) {
    if (!confirm('Are you sure you want to delete this progress update?')) {
        return;
    }

    try {
        const response = await fetch(`/api/tasks/${taskId}/progress/${progressId}`, {
            method: 'DELETE'
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'Failed to delete progress update' }));
            throw new Error(errorData.detail || 'Failed to delete progress update');
        }

        // Remove the progress item from the DOM
        const progressItem = document.getElementById(`progress-${progressId}`);
        if (progressItem) {
            progressItem.remove();
        }

        // Update the count in the header
        const progressList = document.getElementById(`progress-list-${taskId}`);
        const headerCount = document.querySelector(`#progress-content-${taskId}`).previousElementSibling.querySelector('span[style*="text-muted"]');
        if (headerCount) {
            const currentCount = parseInt(headerCount.textContent.replace(/[()]/g, '')) || 0;
            const newCount = Math.max(0, currentCount - 1);
            headerCount.textContent = `(${newCount})`;
        }

        // If no more progress items, show empty state
        if (progressList && progressList.children.length === 0) {
            progressList.innerHTML = `
                <div class="progress-empty" id="progress-empty-${taskId}">
                    <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
                    </svg>
                    <p>No progress updates yet. Start logging your work!</p>
                </div>
            `;
        }

        console.log('Progress update deleted successfully');

    } catch (error) {
        console.error('Error deleting progress update:', error);
        alert(`Error: ${error.message}`);
    }
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

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
                user_type: 'employee',
                user_id: currentEmployeeId,
                user_name: currentEmployeeName
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

    // Check if this is our own message (avoid duplicates from optimistic update)
    const isOwnMessage = (data.sender_type === 'employee' && data.sender_name === currentEmployeeName);
    
    // If it's our own message, look for the temporary optimistic message and replace it
    if (isOwnMessage) {
        const tempMessages = chatMessages.querySelectorAll('[data-temp-id]');
        for (const tempMsg of tempMessages) {
            const tempText = tempMsg.getAttribute('data-message-text');
            if (tempText === data.message) {
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
    
    // Check for duplicate: look for recent message with same text (within last 3 messages)
    const existingMessages = Array.from(chatMessages.querySelectorAll('.chat-message')).slice(-3);
    const isDuplicate = existingMessages.some(msg => {
        const msgText = msg.querySelector('.chat-text').textContent.trim();
        const msgSender = msg.querySelector('.chat-sender').textContent.trim();
        
        // Check if it's the same message (same text and sender)
        if (isOwnMessage) {
            return msgText === data.message && (msgSender === 'You' || msgSender.includes('(You)'));
        } else {
            return msgText === data.message && msgSender.includes(data.sender_name);
        }
    });

    if (isDuplicate) {
        console.log('Duplicate message detected, skipping');
        return;
    }

    // Create message element
    const messageDiv = document.createElement('div');
    const messageClass = isOwnMessage ? 'chat-message employee' : 'chat-message manager';

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

// Initialize WebSocket when page loads
document.addEventListener('DOMContentLoaded', function() {
    // Only initialize if we have the required variables
    if (typeof tenderId !== 'undefined' && typeof currentEmployeeId !== 'undefined' && typeof currentEmployeeName !== 'undefined') {
        initWebSocket();
    } else {
        console.error('Missing required variables for WebSocket connection');
    }
});

// Clean up WebSocket on page unload
window.addEventListener('beforeunload', function() {
    closeWebSocket();
});
