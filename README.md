# DDoS Detection System

## Overview

The DDoS Detection System is an advanced, real-time network traffic monitoring and Distributed Denial of Service (DDoS) attack detection platform powered by Long Short-Term Memory Recurrent Neural Networks (LSTM-RNN). This project leverages machine learning to identify anomalous traffic patterns indicative of DDoS attacks, offering a user-friendly web interface for monitoring, testing, and managing the system. Designed for network administrators and security professionals, it provides a robust toolset to safeguard network infrastructure against DDoS threats.

## Project Structure

The project directory is organized as follows:


### Key Directories and Files

- `data/`: Stores datasets (e.g., CIC-DDoS2019 or synthetic data) for training and testing the LSTM-RNN model.
- `logs/`: Contains log files, such as `prediction.log`, for tracking predictions and system activity.
- `model/`: Holds trained machine learning models or related files generated by `train_model.py`.
- `templates/`: Includes HTML templates (`index.html` and `monitor.html`) for the web interface.
- `attack.py`: Script for simulating DDoS attack traffic to test detection capabilities.
- `dataloader.py`: Manages loading and preprocessing of traffic data for the model.
- `main.py`: Likely the primary script for orchestrating system components or running the application.
- `prediction.log`: Logs real-time predictions and system status updates.
- `prediction.py`: Implements logic for real-time DDoS detection using the trained model.
- `requirements.txt`: Lists Python dependencies (e.g., Flask, TensorFlow/Keras) required to run the project.
- `server.py`: Runs the Flask web server to host the web interface and handle API endpoints.
- `test/`: Directory for unit tests or testing scripts to validate system functionality.
- `train_model.py`: Script for training the LSTM-RNN model on traffic data.

## Functionality

### 1. Real-Time Traffic Monitoring

**Purpose:** Continuously monitors network traffic to provide real-time insights into request rates.

**Implementation:**

- `server.py` and `prediction.py` fetch traffic data via endpoints like `/status` and `/total_count`.
- The `trafficChart` in the `monitor.html` template visualizes requests per second (RPS) using Chart.js.

**Features:**

- Displays current RPS and peak RPS in the monitoring dashboard’s stat boxes.
- Updates every 500 milliseconds for near-real-time visualization.
- Plots traffic trends on a line chart with time (seconds) on the x-axis and request count on the y-axis, dynamically adjusting scales based on peak values.

### 2. DDoS Detection Using LSTM-RNN

**Purpose:** Identifies DDoS attacks by detecting anomalous traffic patterns.

**Implementation:**

- `train_model.py` trains the LSTM-RNN model using historical data from the `data/` directory.
- `prediction.py` applies the model to real-time traffic, analyzing patterns via `server.py` endpoints.

**Features:**

- Compares current traffic against trained model predictions to detect anomalies.
- Updates the Current Traffic Status in the monitoring dashboard:
  - "Normal Traffic" (green background).
  - "DDoS ATTACK DETECTED" (red background with pulsing animation).
- Displays detection confidence via a bar chart showing probabilities for normal vs. attack traffic, updated with `/status` data.

### 3. Web Interface

**Purpose:** Provides an intuitive interface for system interaction and monitoring.

**Implementation:**

- Located in `templates/` with two key HTML files:
  - `index.html`: Main dashboard for system overview and actions.
  - `monitor.html`: Detailed monitoring dashboard with charts and stats.
- Served via `server.py` using Flask.

**Features:**

- **Main Dashboard (`index.html`):**
  - **System Status:** Shows "Detection engine is active" with a green indicator.
  - **Current Traffic:** Displays requests per second, updated every 2 seconds via `/total_count`.
  - **Action Cards:**
    - Traffic Monitor: Links to the monitoring dashboard (`/monitor`).
    - Test Detection: Triggers a simulated attack via `runTest()` and `/test-attack`.
    - Documentation: Placeholder alert for future documentation (`showDocs()`).

- **Monitoring Dashboard (`monitor.html`):**
  - **Stats Container:** Shows real-time metrics (current RPS, peak RPS, attack detections).
  - **Charts:**
    - Traffic Chart: Line chart of RPS over time.
    - Detection Confidence: Bar chart of normal vs. attack probabilities.
    - Total Requests: Line chart of cumulative requests over time.
  - **Actions:**
    - Download Traffic Logs: Retrieves logs via `/download-logs`.
    - Reset Stats: Resets all metrics and charts to zero.

### 4. Simulation and Testing

**Purpose:** Enables testing of detection capabilities with simulated DDoS attacks.

**Implementation:**

- `attack.py` generates simulated attack traffic.
- Triggered via the "Test Detection" button in `index.html`, sending a POST request to `/test-attack`.

**Features:**

- Prompts user confirmation before initiating a test.
- Updates dashboard metrics and charts to reflect increased traffic and detection results.

### 5. Model Performance Monitoring

**Purpose:** Tracks and displays LSTM-RNN model performance metrics.

**Implementation:**

- The `loadModelMetrics()` function in the monitoring dashboard fetches data from `/model-metrics`.
- Metrics are calculated during training or inference and served via `server.py`.

**Features:**

- Displays accuracy, precision, recall, and F1 score in a grid, updated on page load.
- Values are shown as percentages (e.g., 95.23%) for easy interpretation.

### 6. Logging and Data Management

**Purpose:** Records system activity and manages data for analysis.

**Implementation:**

- Logs are stored in `logs/`, with `prediction.log` capturing prediction outcomes.
- `dataloader.py` preprocesses data from `data/` for training and testing.

**Features:**

- Logs real-time predictions and system status for debugging and review.
- Allows downloading logs via the monitoring dashboard’s "Download Traffic Logs" button.

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Git (for cloning the repository)
- Internet connection (for installing dependencies)

### Installation
1. Clone the repository:
   git clone <repository-url>
   cd ddos-detection-system
2. Install dependencies:
    pip install -r requirements.txt
3. Prepare data:
    Place traffic data in data/.
    Data should be preprocessed using dataloader.py.

4. Train the Model
    Run the training script:
    python train_model.py

5. Run the Server/Start the Flask web server:
    python server.py
    Open a web browser and navigate to http://localhost:5000 to access the interface.

Usage

1. Access the Main Dashboard:
Visit http://localhost:5000 to load index.html.
View system status, current traffic, and action options (Traffic Monitor, Test Detection, Documentation).
2. Monitor Traffic:
Click "Open Dashboard" to access the monitoring interface (/monitor).
Observe real-time charts (traffic, confidence, total requests) and stats (RPS, peak RPS, detections).
3. Test Detection:
Select "Run Test" on the main dashboard.
Confirm the prompt to simulate a DDoS attack, then monitor the system’s response in the dashboard.
4. Download Logs:
Click "Download Traffic Logs" in the monitoring dashboard to retrieve prediction.log or other logs.
5. Reset Statistics:
Press "Reset Stats" in the monitoring dashboard to clear all metrics and charts, resetting to zero.