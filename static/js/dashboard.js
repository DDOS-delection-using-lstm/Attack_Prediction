// Dashboard functionality
let trafficChart, confidenceChart, requestCountChart;
let networkVisualizer;
let peakRps = 0;
let detectionCount = 0;
let totalRequests = 0;
let lastStatus = "Normal Traffic";
const MAX_CHART_POINTS = 120;

function fetchData() {
    fetch("/status")
        .then(response => response.json())
        .then(data => {
            updateDashboardWithData(data);
        })
        .catch(error => {
            console.error("Error fetching status data:", error);
            updateStatusIndicator("Error", "error");
            setTimeout(fetchData, 5000);
        });
}

function updateDashboardWithData(data) {
    // Update numeric stats
    const currentRps = data.requests_per_second || 0;
    if (currentRps > peakRps) peakRps = currentRps;
    totalRequests = data.total_requests || 0;
    
    document.getElementById("currentRps").textContent = currentRps > 0 ? currentRps.toLocaleString() : "No Traffic Yet";
    document.getElementById("peakRps").textContent = peakRps > 0 ? peakRps.toLocaleString() : "No Traffic Yet";
    document.getElementById("totalRequestsCounter").textContent = totalRequests > 0 ? totalRequests.toLocaleString() : "No Traffic Yet";

    // Update status
    const status = data.status || "Normal Traffic";
    updateStatusIndicator(status, status === "Normal Traffic" ? "normal" : "alert");

    // Update charts
    updateCharts(data);
}

function updateStatusIndicator(status, type) {
    const statusIndicator = document.getElementById("statusIndicator");
    statusIndicator.textContent = status;
    statusIndicator.className = "badge";
    
    switch (type) {
        case "normal":
            statusIndicator.classList.add("bg-success");
            break;
        case "alert":
            statusIndicator.classList.add("bg-danger");
            break;
        case "warning":
            statusIndicator.classList.add("bg-warning");
            break;
        case "error":
            statusIndicator.classList.add("bg-secondary");
            break;
    }
}

function updateCharts(data) {
    const timestamp = new Date();
    
    if (trafficChart) {
        trafficChart.data.labels.push(timestamp);
        trafficChart.data.datasets[0].data.push(data.requests_per_second || 0);
        
        if (trafficChart.data.labels.length > MAX_CHART_POINTS) {
            trafficChart.data.labels.shift();
            trafficChart.data.datasets[0].data.shift();
        }
        
        trafficChart.update();
    }
    
    if (requestCountChart) {
        requestCountChart.data.labels.push(timestamp);
        requestCountChart.data.datasets[0].data.push(totalRequests);
        
        if (requestCountChart.data.labels.length > MAX_CHART_POINTS) {
            requestCountChart.data.labels.shift();
            requestCountChart.data.datasets[0].data.shift();
        }
        
        requestCountChart.update();
    }
}

function initializeCharts() {
    const trafficCtx = document.getElementById("trafficChart")?.getContext("2d");
    const requestCountCtx = document.getElementById("requestCountChart")?.getContext("2d");
    
    if (trafficCtx) {
        trafficChart = new Chart(trafficCtx, {
            type: "line",
            data: {
                labels: [],
                datasets: [{
                    label: "Requests per Second",
                    data: [],
                    borderColor: "#3498db",
                    fill: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: "time",
                        time: { unit: "second" },
                        title: { display: true, text: "Time" }
                    },
                    y: {
                        title: { display: true, text: "RPS" },
                        beginAtZero: true
                    }
                }
            }
        });
    }
    
    if (requestCountCtx) {
        requestCountChart = new Chart(requestCountCtx, {
            type: "line",
            data: {
                labels: [],
                datasets: [{
                    label: "Total Requests",
                    data: [],
                    borderColor: "#9b59b6",
                    fill: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: "time",
                        time: { unit: "second" },
                        title: { display: true, text: "Time" }
                    },
                    y: {
                        title: { display: true, text: "Requests" },
                        beginAtZero: true
                    }
                }
            }
        });
    }
}

// Initialize dashboard
document.addEventListener("DOMContentLoaded", () => {
    initializeCharts();
    fetchData();
    setInterval(fetchData, 2000);
}); 