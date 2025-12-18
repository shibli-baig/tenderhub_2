/**
 * BidSuite Analytics Dashboard - Chart Rendering and Interactions
 * Handles ApexCharts initialization, data visualization, and user interactions
 */

// Global chart instances
let statusDonutChart = null;
let statusTimelineChart = null;
let valueBarChart = null;
let conversionFunnelChart = null;
let stageFunnelChart = null;
let stageHeatmapChart = null;

// Current active time period
let currentTimePeriod = '30'; // Default to 30 days

// Chart theme configuration
const chartTheme = {
    colors: {
        primary: '#6366f1',
        success: '#22c55e',
        warning: '#f97316',
        danger: '#ef4444',
        neutral: '#64748b',
        purple: '#a855f7',
        cyan: '#06b6d4',
        pink: '#ec4899'
    },
    fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
    fontSize: '13px',
    darkMode: false
};

// Initialize dark mode state
function initDarkMode() {
    chartTheme.darkMode = document.body.classList.contains('dark-mode');
}

// Get common chart options based on current theme
function getCommonChartOptions() {
    return {
        chart: {
            fontFamily: chartTheme.fontFamily,
            toolbar: {
                show: true,
                tools: {
                    download: true,
                    selection: false,
                    zoom: false,
                    zoomin: false,
                    zoomout: false,
                    pan: false,
                    reset: false
                }
            },
            background: 'transparent',
            foreColor: chartTheme.darkMode ? '#e5e7eb' : '#6b7280'
        },
        grid: {
            borderColor: chartTheme.darkMode ? '#374151' : '#e5e7eb',
            strokeDashArray: 4
        },
        tooltip: {
            theme: chartTheme.darkMode ? 'dark' : 'light',
            style: {
                fontSize: chartTheme.fontSize,
                fontFamily: chartTheme.fontFamily
            }
        },
        legend: {
            fontFamily: chartTheme.fontFamily,
            fontSize: chartTheme.fontSize,
            labels: {
                colors: chartTheme.darkMode ? '#e5e7eb' : '#6b7280'
            }
        }
    };
}

/**
 * Section 2: Tenders Analytics Charts
 */

// 1. Status Donut Chart
function renderStatusDonutChart(data) {
    const container = document.getElementById('statusDonutChart');
    if (!container) return;

    // Check if ApexCharts is available
    if (typeof ApexCharts === 'undefined') {
        console.error('❌ ApexCharts library not loaded yet');
        container.innerHTML = '<div style="display: flex; align-items: center; justify-content: center; height: 350px; color: #ef4444; text-align: center;"><div><div style="font-size: 2rem; margin-bottom: 0.5rem;">⚠️</div><div>ApexCharts library failed to load</div><div style="font-size: 0.875rem; margin-top: 0.5rem;">Please check your internet connection</div></div></div>';
        return;
    }

    // Remove loading spinner
    const loadingDiv = container.querySelector('.chart-loading');
    if (loadingDiv) loadingDiv.remove();

    // Destroy existing chart
    if (statusDonutChart) {
        statusDonutChart.destroy();
    }

    const series = [
        data.favorites_count || 0,
        data.shortlisted_count || 0,
        data.rejected_count || 0,
        data.dumped_count || 0,
        data.awarded_count || 0
    ];

    // Check if all values are zero
    const totalCount = series.reduce((a, b) => a + b, 0);
    if (totalCount === 0) {
        container.innerHTML = '<div style="display: flex; align-items: center; justify-content: center; height: 350px; color: #9ca3af; text-align: center;"><div><div style="font-size: 2rem; margin-bottom: 0.5rem;">📊</div><div>No tender data available</div></div></div>';
        return;
    }

    const options = {
        ...getCommonChartOptions(),
        series: series,
        chart: {
            type: 'donut',
            height: 350,
            ...getCommonChartOptions().chart
        },
        labels: ['Favourited', 'Shortlisted', 'Rejected', 'Cancelled/Lost', 'Awarded'],
        colors: [
            '#6366f1',  // Primary blue for Favourited
            '#f59e0b',  // Amber for Shortlisted
            '#ef4444',  // Red for Rejected
            '#64748b',  // Slate for Cancelled/Lost
            '#22c55e'   // Green for Awarded
        ],
        plotOptions: {
            pie: {
                donut: {
                    size: '65%',
                    labels: {
                        show: true,
                        name: {
                            show: true,
                            fontSize: '18px',
                            fontWeight: 600
                        },
                        value: {
                            show: true,
                            fontSize: '24px',
                            fontWeight: 700,
                            formatter: function(val) {
                                return val;
                            }
                        },
                        total: {
                            show: true,
                            label: 'Total Tenders',
                            fontSize: '14px',
                            fontWeight: 500,
                            color: chartTheme.darkMode ? '#e5e7eb' : '#6b7280',
                            formatter: function(w) {
                                return w.globals.seriesTotals.reduce((a, b) => a + b, 0);
                            }
                        }
                    }
                }
            }
        },
        dataLabels: {
            enabled: false
        },
        legend: {
            position: 'bottom',
            horizontalAlign: 'center',
            ...getCommonChartOptions().legend
        },
        responsive: [{
            breakpoint: 768,
            options: {
                chart: {
                    height: 300
                },
                legend: {
                    position: 'bottom'
                }
            }
        }]
    };

    statusDonutChart = new ApexCharts(container, options);
    statusDonutChart.render();
}

// 2. Status Timeline Chart
function renderStatusTimelineChart(data) {
    const container = document.getElementById('statusTimelineChart');
    if (!container) return;

    // Check if ApexCharts is available
    if (typeof ApexCharts === 'undefined') {
        console.error('❌ ApexCharts library not loaded yet');
        container.innerHTML = '<div style="display: flex; align-items: center; justify-content: center; height: 350px; color: #ef4444; text-align: center;"><div><div style="font-size: 2rem; margin-bottom: 0.5rem;">⚠️</div><div>ApexCharts library failed to load</div><div style="font-size: 0.875rem; margin-top: 0.5rem;">Please check your internet connection</div></div></div>';
        return;
    }

    // Remove loading spinner
    const loadingDiv = container.querySelector('.chart-loading');
    if (loadingDiv) loadingDiv.remove();

    // Destroy existing chart
    if (statusTimelineChart) {
        statusTimelineChart.destroy();
    }

    // Prepare timeline data (last 30 days)
    const timelineData = data.timeline_data || [];

    if (timelineData.length === 0) {
        container.innerHTML = '<div style="display: flex; align-items: center; justify-content: center; height: 350px; color: #9ca3af; text-align: center;"><div><div style="font-size: 2rem; margin-bottom: 0.5rem;">📈</div><div>No timeline data available</div></div></div>';
        return;
    }

    const series = [
        {
            name: 'Favourited',
            data: timelineData.map(d => ({ x: d.date, y: d.favorites || 0 }))
        },
        {
            name: 'Shortlisted',
            data: timelineData.map(d => ({ x: d.date, y: d.shortlisted || 0 }))
        },
        {
            name: 'Awarded',
            data: timelineData.map(d => ({ x: d.date, y: d.awarded || 0 }))
        }
    ];

    const options = {
        ...getCommonChartOptions(),
        series: series,
        chart: {
            type: 'area',
            height: 350,
            stacked: false,
            ...getCommonChartOptions().chart
        },
        colors: ['#6366f1', '#f59e0b', '#22c55e'],
        dataLabels: {
            enabled: false
        },
        stroke: {
            curve: 'smooth',
            width: 2
        },
        fill: {
            type: 'gradient',
            gradient: {
                opacityFrom: 0.6,
                opacityTo: 0.1,
                stops: [0, 100]
            }
        },
        xaxis: {
            type: 'datetime',
            labels: {
                datetimeUTC: false,
                format: 'dd MMM',
                style: {
                    colors: chartTheme.darkMode ? '#e5e7eb' : '#6b7280',
                    fontSize: '12px'
                }
            }
        },
        yaxis: {
            title: {
                text: 'Number of Tenders',
                style: {
                    color: chartTheme.darkMode ? '#e5e7eb' : '#6b7280',
                    fontSize: '13px',
                    fontWeight: 500
                }
            },
            labels: {
                style: {
                    colors: chartTheme.darkMode ? '#e5e7eb' : '#6b7280'
                }
            }
        },
        legend: {
            position: 'top',
            horizontalAlign: 'left',  // Move legend to left to avoid overlap with toolbar menu
            offsetY: 0,
            ...getCommonChartOptions().legend
        },
        responsive: [{
            breakpoint: 768,
            options: {
                chart: {
                    height: 300
                }
            }
        }]
    };

    statusTimelineChart = new ApexCharts(container, options);
    statusTimelineChart.render();
}

// 3. Value Bar Chart
function renderValueBarChart(data) {
    const container = document.getElementById('valueBarChart');
    if (!container) return;

    // Check if ApexCharts is available
    if (typeof ApexCharts === 'undefined') {
        console.error('❌ ApexCharts library not loaded yet');
        container.innerHTML = '<div style="display: flex; align-items: center; justify-content: center; height: 350px; color: #ef4444; text-align: center;"><div><div style="font-size: 2rem; margin-bottom: 0.5rem;">⚠️</div><div>ApexCharts library failed to load</div><div style="font-size: 0.875rem; margin-top: 0.5rem;">Please check your internet connection</div></div></div>';
        return;
    }

    // Remove loading spinner
    const loadingDiv = container.querySelector('.chart-loading');
    if (loadingDiv) loadingDiv.remove();

    // Destroy existing chart
    if (valueBarChart) {
        valueBarChart.destroy();
    }

    const series = [{
        name: 'Estimated Value (₹)',
        data: [
            data.shortlisted_value || 0,
            data.awarded_value || 0,
            data.dumped_value || 0
        ]
    }];

    // Check if all values are zero
    const totalValue = series[0].data.reduce((a, b) => a + b, 0);
    if (totalValue === 0) {
        container.innerHTML = '<div style="display: flex; align-items: center; justify-content: center; height: 350px; color: #9ca3af; text-align: center;"><div><div style="font-size: 2rem; margin-bottom: 0.5rem;">💰</div><div>No value data available</div></div></div>';
        return;
    }

    const options = {
        ...getCommonChartOptions(),
        series: series,
        chart: {
            type: 'bar',
            height: 350,
            ...getCommonChartOptions().chart
        },
        colors: ['#6366f1'],
        plotOptions: {
            bar: {
                horizontal: true,
                borderRadius: 8,
                dataLabels: {
                    position: 'center'  // Auto-positions labels inside bars when space available
                }
            }
        },
        dataLabels: {
            enabled: true,
            formatter: function(val) {
                return '₹' + formatCurrency(val);
            },
            // No offsetX - let ApexCharts auto-position based on bar size
            style: {
                fontSize: '12px',
                fontWeight: 600,
                colors: ['#fff']  // White text for better contrast on colored bars
            }
        },
        xaxis: {
            categories: ['Shortlisted', 'Awarded', 'Cancelled/Lost'],
            labels: {
                formatter: function(val) {
                    return '₹' + formatCurrency(val);
                },
                style: {
                    colors: chartTheme.darkMode ? '#e5e7eb' : '#6b7280'
                }
            }
        },
        yaxis: {
            labels: {
                style: {
                    colors: chartTheme.darkMode ? '#e5e7eb' : '#6b7280',
                    fontSize: '13px',
                    fontWeight: 500
                }
            }
        },
        tooltip: {
            y: {
                formatter: function(val) {
                    return '₹' + formatCurrency(val);
                }
            }
        },
        responsive: [{
            breakpoint: 768,
            options: {
                chart: {
                    height: 300
                },
                plotOptions: {
                    bar: {
                        horizontal: false
                    }
                }
            }
        }]
    };

    valueBarChart = new ApexCharts(container, options);
    valueBarChart.render();
}

// 4. Conversion Funnel Chart
function renderConversionFunnelChart(data) {
    const container = document.getElementById('conversionFunnelChart');
    if (!container) return;

    // Check if ApexCharts is available
    if (typeof ApexCharts === 'undefined') {
        console.error('❌ ApexCharts library not loaded yet');
        container.innerHTML = '<div style="display: flex; align-items: center; justify-content: center; height: 350px; color: #ef4444; text-align: center;"><div><div style="font-size: 2rem; margin-bottom: 0.5rem;">⚠️</div><div>ApexCharts library failed to load</div><div style="font-size: 0.875rem; margin-top: 0.5rem;">Please check your internet connection</div></div></div>';
        return;
    }

    // Remove loading spinner
    const loadingDiv = container.querySelector('.chart-loading');
    if (loadingDiv) loadingDiv.remove();

    // Destroy existing chart
    if (conversionFunnelChart) {
        conversionFunnelChart.destroy();
    }

    const series = [
        data.favorites_count || 0,
        data.shortlisted_count || 0,
        data.awarded_count || 0
    ];

    // Check if all values are zero
    const totalCount = series.reduce((a, b) => a + b, 0);
    if (totalCount === 0) {
        container.innerHTML = '<div style="display: flex; align-items: center; justify-content: center; height: 350px; color: #9ca3af; text-align: center;"><div><div style="font-size: 2rem; margin-bottom: 0.5rem;">🔄</div><div>No conversion data available</div></div></div>';
        return;
    }

    const options = {
        ...getCommonChartOptions(),
        series: [{
            name: 'Tender Conversion Funnel',
            data: series
        }],
        chart: {
            type: 'bar',
            height: 350,
            ...getCommonChartOptions().chart
        },
        plotOptions: {
            bar: {
                borderRadius: 0,
                horizontal: true,
                distributed: true,
                barHeight: '80%',
                isFunnel: true
            }
        },
        colors: [
            '#6366f1',  // Blue for Favourited
            '#f59e0b',  // Amber for Shortlisted
            '#22c55e'   // Green for Awarded
        ],
        dataLabels: {
            enabled: true,
            formatter: function(val, opt) {
                return opt.w.globals.labels[opt.dataPointIndex] + ': ' + val;
            },
            dropShadow: {
                enabled: false
            },
            style: {
                fontSize: '13px',
                fontWeight: 600,
                colors: ['#fff']
            }
        },
        xaxis: {
            categories: ['Favourited', 'Shortlisted', 'Awarded']
        },
        legend: {
            show: false
        },
        responsive: [{
            breakpoint: 768,
            options: {
                chart: {
                    height: 300
                }
            }
        }]
    };

    conversionFunnelChart = new ApexCharts(container, options);
    conversionFunnelChart.render();
}

/**
 * Section 3: Stage Analytics Charts
 */

// 5. Stage Funnel Chart (Overview)
function renderStageFunnelChart(data) {
    const container = document.getElementById('stageFunnelChart');
    if (!container) return;

    // Destroy existing chart
    if (stageFunnelChart) {
        stageFunnelChart.destroy();
    }

    // Extract counts per stage from stage_breakdown
    const stageBreakdown = data.stage_breakdown || {};
    const stageCounts = [
        getTotalForStage(stageBreakdown.step1 || {}),
        getTotalForStage(stageBreakdown.step2 || {}),
        getTotalForStage(stageBreakdown.step3 || {}),
        getTotalForStage(stageBreakdown.step4 || {}),
        getTotalForStage(stageBreakdown.step5 || {}),
        getTotalForStage(stageBreakdown.step6 || {})
    ];

    const options = {
        ...getCommonChartOptions(),
        series: [{
            name: 'Tenders in Stage',
            data: stageCounts
        }],
        chart: {
            type: 'bar',
            height: 400,
            ...getCommonChartOptions().chart
        },
        plotOptions: {
            bar: {
                borderRadius: 0,
                horizontal: true,
                distributed: true,
                barHeight: '85%',
                isFunnel: true
            }
        },
        colors: [
            chartTheme.colors.primary,
            chartTheme.colors.purple,
            chartTheme.colors.cyan,
            chartTheme.colors.warning,
            chartTheme.colors.pink,
            chartTheme.colors.success
        ],
        dataLabels: {
            enabled: true,
            formatter: function(val, opt) {
                return opt.w.globals.labels[opt.dataPointIndex] + ': ' + val;
            },
            dropShadow: {
                enabled: false
            },
            style: {
                fontSize: '13px',
                fontWeight: 600,
                colors: ['#fff']
            }
        },
        xaxis: {
            categories: [
                'Pre-Bid Meeting',
                'Proposal Submission',
                'Technical Proposal Opening',
                'Financial Proposal',
                'Further Negotiations',
                'Tender Awarded'
            ]
        },
        legend: {
            show: false
        },
        responsive: [{
            breakpoint: 768,
            options: {
                chart: {
                    height: 350
                },
                dataLabels: {
                    style: {
                        fontSize: '11px'
                    }
                }
            }
        }]
    };

    stageFunnelChart = new ApexCharts(container, options);
    stageFunnelChart.render();
}

// 6. Stage Heatmap Chart (Overview)
function renderStageHeatmapChart(data) {
    const container = document.getElementById('stageHeatmapChart');
    if (!container) return;

    // Destroy existing chart
    if (stageHeatmapChart) {
        stageHeatmapChart.destroy();
    }

    const stageBreakdown = data.stage_breakdown || {};

    // Prepare heatmap data: rows are stages, columns are statuses
    const stages = [
        'Pre-Bid Meeting',
        'Proposal Submission',
        'Tech Opening',
        'Financial Proposal',
        'Negotiations',
        'Awarded'
    ];

    // Common statuses across stages
    const statuses = ['Pending', 'In Progress', 'Completed', 'Attended', 'Not Attended'];

    // Build series data
    const series = stages.map((stage, idx) => {
        const stepKey = `step${idx + 1}`;
        const stepData = stageBreakdown[stepKey] || {};

        return {
            name: stage,
            data: statuses.map(status => stepData[status] || 0)
        };
    });

    const options = {
        ...getCommonChartOptions(),
        series: series,
        chart: {
            type: 'heatmap',
            height: 400,
            ...getCommonChartOptions().chart
        },
        dataLabels: {
            enabled: true,
            style: {
                colors: ['#fff'],
                fontSize: '12px',
                fontWeight: 600
            }
        },
        colors: [chartTheme.colors.primary],
        xaxis: {
            categories: statuses,
            labels: {
                style: {
                    colors: chartTheme.darkMode ? '#e5e7eb' : '#6b7280',
                    fontSize: '12px'
                }
            }
        },
        yaxis: {
            labels: {
                style: {
                    colors: chartTheme.darkMode ? '#e5e7eb' : '#6b7280',
                    fontSize: '12px'
                }
            }
        },
        plotOptions: {
            heatmap: {
                radius: 4,
                enableShades: true,
                shadeIntensity: 0.5,
                colorScale: {
                    ranges: [
                        { from: 0, to: 0, color: chartTheme.darkMode ? '#374151' : '#f3f4f6', name: 'None' },
                        { from: 1, to: 3, color: chartTheme.colors.success + '80', name: 'Low' },
                        { from: 4, to: 7, color: chartTheme.colors.warning + '80', name: 'Medium' },
                        { from: 8, to: 999, color: chartTheme.colors.danger + '80', name: 'High' }
                    ]
                }
            }
        },
        responsive: [{
            breakpoint: 768,
            options: {
                chart: {
                    height: 350
                },
                dataLabels: {
                    style: {
                        fontSize: '10px'
                    }
                }
            }
        }]
    };

    stageHeatmapChart = new ApexCharts(container, options);
    stageHeatmapChart.render();
}

/**
 * Utility Functions
 */

// Format currency values
function formatCurrency(value) {
    if (value >= 10000000) { // 1 crore
        return (value / 10000000).toFixed(2) + ' Cr';
    } else if (value >= 100000) { // 1 lakh
        return (value / 100000).toFixed(2) + ' L';
    } else if (value >= 1000) {
        return (value / 1000).toFixed(2) + ' K';
    }
    return value.toFixed(2);
}

// Get total count for a stage
function getTotalForStage(stageData) {
    return Object.values(stageData).reduce((sum, count) => sum + count, 0);
}

// Time period filtering
function handleTimePeriodChange(period) {
    currentTimePeriod = period;

    // Update active button
    document.querySelectorAll('.time-period-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.period === period) {
            btn.classList.add('active');
        }
    });

    // Fetch new data and re-render charts
    if (period === 'custom') {
        // Show custom date range picker
        showCustomDateRangePicker();
    } else {
        // Reload page with time period parameter
        const url = new URL(window.location);
        url.searchParams.set('period', period);
        window.location.href = url.toString();
    }
}

// Custom date range picker
function showCustomDateRangePicker() {
    const startDate = prompt('Enter start date (YYYY-MM-DD):');
    const endDate = prompt('Enter end date (YYYY-MM-DD):');

    if (startDate && endDate) {
        const url = new URL(window.location);
        url.searchParams.set('period', 'custom');
        url.searchParams.set('start_date', startDate);
        url.searchParams.set('end_date', endDate);
        window.location.href = url.toString();
    }
}

// View toggle for Section 3
function toggleStageView(view) {
    // Update active button
    document.querySelectorAll('.view-toggle-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.view === view) {
            btn.classList.add('active');
        }
    });

    // Toggle visibility
    const detailedView = document.getElementById('detailedView');
    const overviewView = document.getElementById('overviewView');

    if (view === 'detailed') {
        detailedView.style.display = 'block';
        overviewView.style.display = 'none';
    } else {
        detailedView.style.display = 'none';
        overviewView.style.display = 'block';

        // Render overview charts if not already rendered
        if (!stageFunnelChart && !stageHeatmapChart) {
            renderOverviewCharts();
        }
    }
}

// Render overview charts (Section 3)
function renderOverviewCharts() {
    // Get data from window object (passed from template)
    const analyticsData = window.analyticsData || {};

    renderStageFunnelChart(analyticsData);
    renderStageHeatmapChart(analyticsData);
}

// Reminder dismissal
async function dismissReminder(reminderId) {
    try {
        const response = await fetch(`/api/reminders/${reminderId}/dismiss`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        if (response.ok) {
            // Remove reminder card from DOM
            const reminderCard = document.querySelector(`[data-reminder-id="${reminderId}"]`);
            if (reminderCard) {
                reminderCard.style.opacity = '0';
                setTimeout(() => reminderCard.remove(), 300);
            }

            // Show success message
            showNotification('Reminder dismissed successfully', 'success');
        } else {
            showNotification('Failed to dismiss reminder', 'error');
        }
    } catch (error) {
        console.error('Error dismissing reminder:', error);
        showNotification('An error occurred', 'error');
    }
}

// Show notification toast
function showNotification(message, type = 'info') {
    // Simple toast notification (you can enhance this)
    const toast = document.createElement('div');
    toast.className = `notification-toast ${type}`;
    toast.textContent = message;
    toast.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 12px 24px;
        background: ${type === 'success' ? '#22c55e' : type === 'error' ? '#ef4444' : '#6366f1'};
        color: white;
        border-radius: 8px;
        font-size: 14px;
        font-weight: 500;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        z-index: 10000;
        animation: slideInRight 0.3s ease;
    `;

    document.body.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'slideOutRight 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

/**
 * Main Initialization Function
 * Called when analytics.js is loaded by the template
 */
function initializeAnalyticsCharts() {
    // Initialize dark mode state
    initDarkMode();

    // Get analytics data from window object (passed from template)
    const analyticsData = window.analyticsData || {};

    // Render Section 2 charts (always visible when section is active)
    renderStatusDonutChart(analyticsData);
    renderStatusTimelineChart(analyticsData);
    renderValueBarChart(analyticsData);
    renderConversionFunnelChart(analyticsData);

    // Attach time period change listeners
    document.querySelectorAll('.time-period-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            handleTimePeriodChange(this.dataset.period);
        });
    });

    // Attach view toggle listeners
    document.querySelectorAll('.view-toggle-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            toggleStageView(this.dataset.view);
        });
    });

    // Listen for dark mode changes
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.attributeName === 'class') {
                initDarkMode();
                // Re-render all charts with new theme
                renderStatusDonutChart(analyticsData);
                renderStatusTimelineChart(analyticsData);
                renderValueBarChart(analyticsData);
                renderConversionFunnelChart(analyticsData);

                // Re-render overview charts if visible
                const overviewView = document.getElementById('overviewView');
                if (overviewView && overviewView.style.display !== 'none') {
                    renderOverviewCharts();
                }
            }
        });
    });

    observer.observe(document.body, {
        attributes: true,
        attributeFilter: ['class']
    });

    console.log('✅ Analytics charts initialized successfully');
}

// Export functions for global access
window.initializeAnalyticsCharts = initializeAnalyticsCharts;
window.dismissReminder = dismissReminder;
window.handleTimePeriodChange = handleTimePeriodChange;
window.toggleStageView = toggleStageView;
