// BidSuite - Main JavaScript File

// Dark Mode Theme Toggle
const THEME_STORAGE_KEY = 'theme';
const themeMediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
let themeToggleButton = null;
let themeToggleLabel = null;

function getStoredTheme() {
    try {
        return localStorage.getItem(THEME_STORAGE_KEY);
    } catch (error) {
        console.warn('Unable to read theme preference from storage.', error);
        return null;
    }
}

function storeTheme(theme) {
    try {
        localStorage.setItem(THEME_STORAGE_KEY, theme);
    } catch (error) {
        console.warn('Unable to persist theme preference.', error);
    }
}

function clearStoredTheme() {
    try {
        localStorage.removeItem(THEME_STORAGE_KEY);
    } catch (error) {
        console.warn('Unable to clear theme preference.', error);
    }
}

function applyTheme(theme, { persist = true, preference = 'user' } = {}) {
    const isDark = theme === 'dark';

    document.body.classList.toggle('dark-mode', isDark);
    document.documentElement.setAttribute('data-theme', theme);

    if (persist) {
        storeTheme(theme);
    }

    if (themeToggleButton) {
        const actionLabel = isDark ? 'Switch to light mode' : 'Switch to dark mode';
        const preferenceSuffix = preference === 'system' ? ' (following system theme)' : '';
        const hint = preference === 'system'
            ? 'Shift+Click to set a manual preference.'
            : 'Shift+Click to return to system theme.';
        const descriptiveLabel = `${actionLabel}${preferenceSuffix}`;
        const tooltipLabel = `${descriptiveLabel} — ${hint.replace(/\.$/, '')}`;

        themeToggleButton.setAttribute('aria-pressed', String(isDark));
        themeToggleButton.dataset.themeState = theme;
        themeToggleButton.dataset.themePreference = preference;
        themeToggleButton.setAttribute('aria-label', descriptiveLabel);
        themeToggleButton.setAttribute('title', tooltipLabel);

        const themeToggleIcon = themeToggleButton.querySelector('[data-theme-toggle-icon]');
        if (themeToggleIcon) {
            themeToggleIcon.textContent = isDark ? '☀️' : '🌙';
        }
    }

    if (themeToggleLabel) {
        const actionLabel = isDark ? 'Switch to light mode' : 'Switch to dark mode';
        const preferenceSuffix = preference === 'system' ? ' (following system theme)' : '';
        const hint = preference === 'system'
            ? 'Shift+Click to set a manual preference.'
            : 'Shift+Click to return to system theme.';
        themeToggleLabel.textContent = `${actionLabel}${preferenceSuffix}. ${hint}`;
    }
}

function toggleTheme() {
    const currentTheme = document.body.classList.contains('dark-mode') ? 'dark' : 'light';
    const nextTheme = currentTheme === 'dark' ? 'light' : 'dark';

    applyTheme(nextTheme, { persist: true, preference: 'user' });
}

function initializeTheme() {
    themeToggleButton = document.querySelector('[data-theme-toggle]');
    themeToggleLabel = document.querySelector('[data-theme-toggle-label]');

    const storedTheme = getStoredTheme();
    const initialTheme = storedTheme || 'light';  // Always default to light mode
    const initialPreference = storedTheme ? 'user' : 'system';

    // Persist the default light theme to prevent system preference from overriding it
    applyTheme(initialTheme, { persist: !storedTheme, preference: initialPreference });

    const handleSystemPreferenceChange = (event) => {
        if (getStoredTheme()) {
            return;
        }

        const themeFromSystem = event.matches ? 'dark' : 'light';
        applyTheme(themeFromSystem, { persist: false, preference: 'system' });
    };

    if (themeMediaQuery.addEventListener) {
        themeMediaQuery.addEventListener('change', handleSystemPreferenceChange);
    } else if (themeMediaQuery.addListener) {
        themeMediaQuery.addListener(handleSystemPreferenceChange);
    }

    if (themeToggleButton) {
        themeToggleButton.addEventListener('click', (event) => {
            if (event.shiftKey) {
                clearStoredTheme();
                const systemTheme = themeMediaQuery.matches ? 'dark' : 'light';
                applyTheme(systemTheme, { persist: false, preference: 'system' });
                return;
            }

            toggleTheme();
        });
    }
}

// Utility Functions
function formatCurrency(amount) {
    if (!amount) return 'N/A';
    return new Intl.NumberFormat('en-IN', {
        style: 'currency',
        currency: 'INR',
        maximumFractionDigits: 0
    }).format(amount);
}

function formatDate(dateString) {
    if (!dateString) return 'N/A';
    try {
        const date = new Date(dateString);
        return date.toLocaleDateString('en-IN', {
            day: '2-digit',
            month: 'short',
            year: 'numeric'
        });
    } catch (e) {
        return dateString;
    }
}

function getDeadlineClass(deadline) {
    if (!deadline) return '';
    
    const now = new Date();
    const deadlineDate = new Date(deadline);
    const diffDays = Math.ceil((deadlineDate - now) / (1000 * 60 * 60 * 24));
    
    if (diffDays < 3) return 'soon';
    if (diffDays < 7) return 'warning';
    return 'safe';
}

function showNotification(message, type = 'success') {
    // Remove existing notifications
    const existing = document.querySelector('.notification');
    if (existing) existing.remove();
    
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    
    // Style the notification
    Object.assign(notification.style, {
        position: 'fixed',
        top: '20px',
        right: '20px',
        padding: '15px 20px',
        borderRadius: '6px',
        color: 'white',
        fontWeight: '600',
        zIndex: '9999',
        minWidth: '300px',
        boxShadow: '0 4px 20px rgba(0,0,0,0.2)',
        transition: 'all 0.3s ease'
    });
    
    if (type === 'success') {
        notification.style.background = '#28a745';
    } else if (type === 'error') {
        notification.style.background = '#dc3545';
    } else {
        notification.style.background = '#17a2b8';
    }
    
    document.body.appendChild(notification);
    
    // Auto remove after 3 seconds
    setTimeout(() => {
        notification.style.transform = 'translateX(100%)';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Form Validation
function validateEmail(email) {
    const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return re.test(email);
}

function validatePassword(password) {
    return password.length >= 6;
}

function validateForm(formId) {
    const form = document.getElementById(formId);
    const inputs = form.querySelectorAll('input[required]');
    let isValid = true;
    
    inputs.forEach(input => {
        const value = input.value.trim();
        let fieldValid = true;
        
        if (!value) {
            fieldValid = false;
        } else if (input.type === 'email' && !validateEmail(value)) {
            fieldValid = false;
        } else if (input.type === 'password' && !validatePassword(value)) {
            fieldValid = false;
        }
        
        // Visual feedback
        if (fieldValid) {
            input.style.borderColor = '#28a745';
        } else {
            input.style.borderColor = '#dc3545';
            isValid = false;
        }
    });
    
    return isValid;
}

// Search and Filter Functions
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function updateSearchResults() {
    const searchParams = new URLSearchParams();
    
    const search = document.getElementById('search')?.value;
    const category = document.getElementById('category')?.value;
    const state = document.getElementById('state')?.value;
    const minValue = document.getElementById('min_value')?.value;
    const maxValue = document.getElementById('max_value')?.value;
    
    if (search) searchParams.set('search', search);
    if (category) searchParams.set('category', category);
    if (state) searchParams.set('state', state);
    if (minValue) searchParams.set('min_value', minValue);
    if (maxValue) searchParams.set('max_value', maxValue);
    
    // Reset to page 1 when searching
    searchParams.set('page', '1');
    
    window.location.href = `/?${searchParams.toString()}`;
}

// Debounced search function
const debouncedSearch = debounce(updateSearchResults, 500);

// Favorites Functions
async function toggleFavorite(tenderId) {
    const favoriteBtn = document.getElementById('favoriteBtn');
    const isFavorited = favoriteBtn.classList.contains('favorited');
    
    try {
        favoriteBtn.disabled = true;
        favoriteBtn.textContent = 'Processing...';
        
        const method = isFavorited ? 'DELETE' : 'POST';
        const formData = new FormData();
        
        const response = await fetch(`/api/favorites/${tenderId}`, {
            method: method,
            body: method === 'POST' ? formData : null
        });
        
        if (response.ok) {
            if (isFavorited) {
                favoriteBtn.classList.remove('favorited');
                favoriteBtn.textContent = '⭐ Add to Favorites';
                showNotification('Removed from favorites');
            } else {
                favoriteBtn.classList.add('favorited');
                favoriteBtn.textContent = '⭐ Favorited';
                showNotification('Added to favorites');
            }
        } else {
            const error = await response.json();
            showNotification(error.detail || 'Error updating favorites', 'error');
        }
    } catch (error) {
        console.error('Error:', error);
        showNotification('Network error occurred', 'error');
    } finally {
        favoriteBtn.disabled = false;
    }
}

async function removeFavorite(tenderId) {
    if (!confirm('Are you sure you want to remove this tender from your favorites?')) {
        return;
    }
    
    try {
        const response = await fetch(`/api/favorites/${tenderId}`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            // Remove the tender card from DOM
            const tenderCard = document.querySelector(`[data-tender-id="${tenderId}"]`);
            if (tenderCard) {
                tenderCard.style.transition = 'all 0.3s ease';
                tenderCard.style.transform = 'scale(0)';
                tenderCard.style.opacity = '0';
                setTimeout(() => tenderCard.remove(), 300);
            }
            showNotification('Removed from favorites');
        } else {
            const error = await response.json();
            showNotification(error.detail || 'Error removing favorite', 'error');
        }
    } catch (error) {
        console.error('Error:', error);
        showNotification('Network error occurred', 'error');
    }
}

// Profile Functions
function toggleProfileEdit() {
    const viewMode = document.getElementById('profileView');
    const editMode = document.getElementById('profileEdit');
    const editBtn = document.getElementById('editProfileBtn');
    
    if (viewMode && editMode) {
        if (editMode.style.display === 'none' || !editMode.style.display) {
            viewMode.style.display = 'none';
            editMode.style.display = 'block';
            editBtn.textContent = 'Cancel';
        } else {
            viewMode.style.display = 'block';
            editMode.style.display = 'none';
            editBtn.textContent = 'Edit Profile';
        }
    }
}

// Logout Function
async function logout() {
    try {
        const response = await fetch('/api/auth/logout', {
            method: 'POST'
        });
        
        if (response.ok) {
            window.location.href = '/';
        }
    } catch (error) {
        console.error('Logout error:', error);
        // Force redirect even if request fails
        window.location.href = '/';
    }
}

// Loading States
function showLoading(element) {
    if (element) {
        element.innerHTML = '<div class="loading">Loading...</div>';
    }
}

function hideLoading(element, originalContent) {
    if (element) {
        element.innerHTML = originalContent;
    }
}

// Tender Card Enhancement
function enhanceTenderCards() {
    const tenderCards = document.querySelectorAll('.tender-card');
    
    tenderCards.forEach(card => {
        // Add click handler for navigation
        card.addEventListener('click', function(e) {
            if (e.target.closest('button') || e.target.closest('a')) {
                return;
            }
            const tenderId = this.getAttribute('data-tender-id');
            if (tenderId) {
                const returnTo = encodeURIComponent(`${window.location.pathname}${window.location.search}`);
                window.location.href = `/tender/${tenderId}?return_to=${returnTo}`;
            }
        });
        
        // Enhance deadline display
        const deadlineElement = card.querySelector('.deadline');
        if (deadlineElement) {
            const deadline = deadlineElement.getAttribute('data-deadline');
            if (deadline) {
                deadlineElement.classList.add(getDeadlineClass(deadline));
            }
        }
        
        // Format currency values
        const valueElement = card.querySelector('.tender-value');
        if (valueElement) {
            const value = parseFloat(valueElement.getAttribute('data-value'));
            if (value) {
                valueElement.textContent = formatCurrency(value);
            }
        }
    });
}

// Initialize search functionality
function initializeSearch() {
    // Don't initialize on procurement page (it has its own filter logic)
    if (window.location.pathname === '/procurement') {
        return;
    }

    const searchInput = document.getElementById('search');
    const categorySelect = document.getElementById('category');
    const stateSelect = document.getElementById('state');
    const minValueInput = document.getElementById('min_value');
    const maxValueInput = document.getElementById('max_value');

    // Add event listeners for search
    if (searchInput) {
        searchInput.addEventListener('input', debouncedSearch);
    }

    if (categorySelect) {
        categorySelect.addEventListener('change', updateSearchResults);
    }

    if (stateSelect) {
        stateSelect.addEventListener('change', updateSearchResults);
    }

    if (minValueInput) {
        minValueInput.addEventListener('input', debouncedSearch);
    }

    if (maxValueInput) {
        maxValueInput.addEventListener('input', debouncedSearch);
    }
}

// Initialize form handling
function initializeForms() {
    // Login form
    const loginForm = document.getElementById('loginForm');
    if (loginForm) {
        loginForm.addEventListener('submit', function(e) {
            if (!validateForm('loginForm')) {
                e.preventDefault();
                showNotification('Please fill all fields correctly', 'error');
            }
        });
    }
    
    // Signup form
    const signupForm = document.getElementById('signupForm');
    if (signupForm) {
        signupForm.addEventListener('submit', function(e) {
            if (!validateForm('signupForm')) {
                e.preventDefault();
                showNotification('Please fill all fields correctly', 'error');
            }
            
            const password = document.getElementById('signup_password')?.value;
            const confirmPassword = document.getElementById('confirm_password')?.value;
            
            if (password !== confirmPassword) {
                e.preventDefault();
                showNotification('Passwords do not match', 'error');
            }
        });
    }
    
    // Profile form
    const profileForm = document.getElementById('profileForm');
    if (profileForm) {
        profileForm.addEventListener('submit', function(e) {
            if (!validateForm('profileForm')) {
                e.preventDefault();
                showNotification('Please fill all required fields', 'error');
            }
        });
    }
}

// Keyboard shortcuts
function initializeKeyboardShortcuts() {
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + K for search focus
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            const searchInput = document.getElementById('search');
            if (searchInput) {
                searchInput.focus();
                searchInput.select();
            }
        }
        
        // Escape to close modals or cancel edit mode
        if (e.key === 'Escape') {
            const editMode = document.getElementById('profileEdit');
            if (editMode && editMode.style.display === 'block') {
                toggleProfileEdit();
            }
        }
    });
}

// Auto-refresh functionality for real-time updates
function initializeAutoRefresh() {
    // Only refresh on home page and dashboard
    const isHomePage = window.location.pathname === '/';
    const isDashboard = window.location.pathname === '/dashboard';
    
    if (isHomePage || isDashboard) {
        // Refresh every 5 minutes
        setInterval(() => {
            // Silent refresh - could be enhanced to only update if new data available
            console.log('Auto-refresh check...');
        }, 5 * 60 * 1000);
    }
}

// Error handling for images
function handleImageError(img) {
    img.style.display = 'none';
}

function getNotificationConfig() {
    const notificationCenter = document.querySelector('.notification-center');
    if (!notificationCenter) {
        return null;
    }

    return {
        center: notificationCenter,
        endpoint: notificationCenter.dataset.endpoint || '/api/notifications',
        markAllEndpoint: notificationCenter.dataset.markAllEndpoint || '/api/notifications/mark-all-read'
    };
}

// Initialize notification badge count
async function initializeNotifications() {
    const config = getNotificationConfig();
    if (!config) {
        return;
    }

    try {
        const response = await fetch(config.endpoint);
        if (!response.ok) return;

        const data = await response.json();
        const badge = document.querySelector('.notification-badge');

        if (badge) {
            const count = data.count || 0;
            badge.textContent = count;
            badge.style.display = count > 0 ? 'flex' : 'none';
        }
    } catch (error) {
        console.error('Error loading notification count:', error);
    }
}

// Initialize everything when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize all functionality
    initializeSearch();
    initializeForms();
    initializeKeyboardShortcuts();
    initializeAutoRefresh();
    initializeNotifications();

    // Enhance existing elements
    enhanceTenderCards();
    
    // Add smooth scrolling to anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Add loading states to buttons
    document.querySelectorAll('form').forEach(form => {
        form.addEventListener('submit', function() {
            const submitBtn = form.querySelector('button[type="submit"]');
            if (submitBtn) {
                const originalText = submitBtn.textContent;
                submitBtn.textContent = 'Processing...';
                submitBtn.disabled = true;
                
                // Re-enable after 10 seconds as fallback
                setTimeout(() => {
                    submitBtn.textContent = originalText;
                    submitBtn.disabled = false;
                }, 10000);
            }
        });
    });
    
    console.log('BidSuite initialized successfully');
});

// Global functions for HTML onclick handlers
window.toggleFavorite = toggleFavorite;
window.removeFavorite = removeFavorite;
window.toggleProfileEdit = toggleProfileEdit;
window.logout = logout;

// Notification Center Functions
async function toggleNotifications(event) {
    event.stopPropagation();
    const dropdown = document.getElementById('notificationDropdown');

    if (dropdown.style.display === 'none' || dropdown.style.display === '') {
        // Fetch notifications from API
        await loadNotifications();
        dropdown.style.display = 'flex';
        // Close dropdown when clicking outside
        setTimeout(() => {
            document.addEventListener('click', closeNotificationsOnClickOutside);
        }, 0);
    } else {
        dropdown.style.display = 'none';
        document.removeEventListener('click', closeNotificationsOnClickOutside);
    }
}

async function loadNotifications() {
    const config = getNotificationConfig();
    if (!config) {
        return;
    }

    try {
        const response = await fetch(config.endpoint);
        if (!response.ok) {
            console.error('Failed to fetch notifications');
            return;
        }

        const data = await response.json();
        const notificationList = document.getElementById('notificationList');
        const badge = document.querySelector('.notification-badge');

        // Update badge count
        if (badge) {
            const count = data.count || 0;
            badge.textContent = count;
            badge.style.display = count > 0 ? 'flex' : 'none';
        }

        // Clear existing notifications
        notificationList.innerHTML = '';

        if (!data.notifications || data.notifications.length === 0) {
            notificationList.innerHTML = '<div class="notification-empty"><p>No notifications yet</p></div>';
            return;
        }

        // Add notifications
        data.notifications.forEach(notification => {
            const notifElement = document.createElement('div');
            notifElement.className = `notification-item ${notification.is_read ? '' : 'unread'}`.trim();
            notifElement.innerHTML = `
                <div class="notification-icon">${notification.icon || '🔔'}</div>
                <div class="notification-content">
                    <div class="notification-title">${notification.title}</div>
                    <div class="notification-message">${notification.message}</div>
                    <div class="notification-time">${notification.time_ago}</div>
                </div>
            `;

            // Add click handler to navigate to tender
            notifElement.addEventListener('click', () => {
                if (notification.link) {
                    window.location.href = notification.link;
                } else if (notification.tender_id) {
                    window.location.href = `/tender/${notification.tender_id}`;
                }
            });

            notificationList.appendChild(notifElement);
        });
    } catch (error) {
        console.error('Error loading notifications:', error);
    }
}

function closeNotificationsOnClickOutside(event) {
    const dropdown = document.getElementById('notificationDropdown');
    const notificationCenter = event.target.closest('.notification-center');
    
    if (!notificationCenter && dropdown) {
        dropdown.style.display = 'none';
        document.removeEventListener('click', closeNotificationsOnClickOutside);
    }
}

async function markAllAsRead() {
    const config = getNotificationConfig();
    if (!config) {
        return;
    }

    try {
        const response = await fetch(config.markAllEndpoint, {
            method: 'POST'
        });

        if (!response.ok) {
            console.error('Failed to mark notifications as read');
            return;
        }

        // Get all unread notification items
        const unreadItems = document.querySelectorAll('.notification-item.unread');

        unreadItems.forEach(item => {
            item.classList.remove('unread');
        });

        // Update badge count
        const badge = document.querySelector('.notification-badge');
        if (badge) {
            badge.textContent = '0';
            badge.style.display = 'none';
        }

        // Reload notifications to show empty state
        await loadNotifications();
    } catch (error) {
        console.error('Error marking notifications as read:', error);
    }
}

// User Menu Dropdown
function toggleUserMenu(event) {
    event.stopPropagation();

    const dropdown = document.getElementById('userMenuDropdown');
    const trigger = event.currentTarget;

    // Close notification dropdown if open
    const notificationDropdown = document.getElementById('notificationDropdown');
    if (notificationDropdown) {
        notificationDropdown.style.display = 'none';
    }

    // Toggle user menu dropdown
    if (dropdown.style.display === 'none' || !dropdown.style.display) {
        dropdown.style.display = 'block';
        trigger.classList.add('active');
    } else {
        dropdown.style.display = 'none';
        trigger.classList.remove('active');
    }
}

// Close dropdowns when clicking outside
document.addEventListener('click', function(event) {
    const userMenu = document.querySelector('.user-menu');
    const userDropdown = document.getElementById('userMenuDropdown');

    if (userMenu && userDropdown && !userMenu.contains(event.target)) {
        userDropdown.style.display = 'none';
        const trigger = userMenu.querySelector('.user-menu-trigger');
        if (trigger) {
            trigger.classList.remove('active');
        }
    }
});

// Initialize theme controls once DOM is ready
initializeTheme();

// Global functions for HTML onclick handlers
window.toggleTheme = toggleTheme;
window.toggleNotifications = toggleNotifications;
window.markAllAsRead = markAllAsRead;
window.toggleUserMenu = toggleUserMenu;
