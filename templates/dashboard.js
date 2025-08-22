// Dashboard JavaScript for CopperQuant

document.addEventListener('DOMContentLoaded', function() {
    // Add event listeners for tabs if they exist
    const tabs = document.querySelectorAll('.tab');
    if (tabs.length > 0) {
        tabs.forEach(tab => {
            tab.addEventListener('click', function() {
                // Remove active class from all tabs
                tabs.forEach(t => t.classList.remove('active'));
                // Add active class to clicked tab
                this.classList.add('active');
                
                // Hide all tab content
                const tabContents = document.querySelectorAll('.tab-content');
                tabContents.forEach(content => content.classList.remove('active'));
                
                // Show the corresponding tab content
                const targetId = this.getAttribute('data-target');
                document.getElementById(targetId).classList.add('active');
            });
        });
    }
    
    // Add refresh button functionality
    const refreshBtn = document.getElementById('refresh-btn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', function() {
            location.reload();
        });
    }
    
    // Add timestamp display
    const timestampElement = document.getElementById('timestamp');
    if (timestampElement) {
        const now = new Date();
        timestampElement.textContent = now.toLocaleString();
    }
    
    // Add chart toggle functionality
    const chartToggles = document.querySelectorAll('.chart-toggle');
    chartToggles.forEach(toggle => {
        toggle.addEventListener('click', function() {
            const chartId = this.getAttribute('data-chart');
            const chartElement = document.getElementById(chartId);
            if (chartElement.style.display === 'none') {
                chartElement.style.display = 'block';
                this.textContent = 'Hide Chart';
            } else {
                chartElement.style.display = 'none';
                this.textContent = 'Show Chart';
            }
        });
    });
    
    // Add signal highlighting
    const signalElement = document.querySelector('.signal');
    if (signalElement) {
        // Add a pulsing effect to the signal
        setInterval(() => {
            signalElement.classList.toggle('pulse');
        }, 1500);
    }
    
    // Add tooltip functionality
    const tooltips = document.querySelectorAll('[data-tooltip]');
    tooltips.forEach(tooltip => {
        tooltip.addEventListener('mouseover', function() {
            const tooltipText = this.getAttribute('data-tooltip');
            const tooltipElement = document.createElement('div');
            tooltipElement.className = 'tooltip';
            tooltipElement.textContent = tooltipText;
            document.body.appendChild(tooltipElement);
            
            const rect = this.getBoundingClientRect();
            tooltipElement.style.top = rect.bottom + 'px';
            tooltipElement.style.left = rect.left + 'px';
        });
        
        tooltip.addEventListener('mouseout', function() {
            const tooltipElement = document.querySelector('.tooltip');
            if (tooltipElement) {
                tooltipElement.remove();
            }
        });
    });
});
