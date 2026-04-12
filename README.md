# CropOracle — Uncertainty-Aware Hybrid Crop AI

CropOracle is a full-stack, machine-learning-powered agricultural platform that predicts the most suitable crop to cultivate based on soil health and climate conditions. 

Unlike traditional platforms that use a single model, CropOracle relies on a **Stacked Hybrid Ensemble** (XGBoost, CatBoost, LightGBM, Random Forest, NGBoost) to deliver **Uncertainty-Aware** predictions and **GenAI-powered** location autofill.

---

## 🏗️ Project Architecture & DevOps Stack

This project is fully containerized and instrumented with a modern DevOps stack designed for local execution and Kubernetes deployment.

### Tech Stack
- **Backend**: Python 3, FastAPI, SQLAlchemy, Google OAuth Authlib
- **Frontend**: Vanilla HTML/CSS/JS (Dark Glassmorphism Design System)
- **Database**: PostgreSQL (Docker) / SQLite (Fallback)
- **Monitoring**: Prometheus & Grafana
- **CI/CD**: Jenkins, Docker, Kubernetes (Minikube)

---

## 🚀 How to Run the Project (Local DevOps)

You need Docker Desktop and Minikube installed.

### 1. The Core App + Database
We use Docker Compose to run the PostgreSQL database and the FastAPI application together.
```bash
# In the root directory:
docker-compose up -d --build
```
- App validates heavily against `.env.local` for local execution.
- Website runs at: `http://localhost:8081` (assuming external port maps to 8081 or 8080 depending on env).

### 2. The Monitoring Stack
Prometheus scrapes the FastAPI `/metrics` endpoint and Node Exporter. Grafana visualizes it.
```bash
# Inside the root directory:
docker-compose -f monitoring/docker-compose.monitoring.yml up -d
```
- **Prometheus**: `http://localhost:9090`
- **Grafana**: `http://localhost:3000` (Login: `admin` / `admin`)

### 3. Kubernetes / Minikube
For Kubernetes-based deployment.
```bash
# Start Minikube
minikube start --driver=docker

# Apply Kubernetes configurations
kubectl apply -f k8s/postgres-pvc.yaml
kubectl apply -f k8s/postgres-deployment.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml

# Create a proxy/tunnel (if needed in a new terminal)
minikube tunnel
```

### 4. Jenkins Pipeline
Start Jenkins using Docker if not already running natively.
```bash
# Get the initial admin password from the docker container (if running via docker)
docker exec jenkins cat /var/jenkins_home/secrets/initialAdminPassword
# Jenkins is mapped to: http://localhost:8082
```

---

## 🌊 Flow from End User to Grafana Metrics

When a user visits the CropOracle web app and clicks around (e.g., clicking the "Predict" button or navigating pages), a precise flow of data triggers changes in your Grafana CPU and Memory graphs:

1. **User Action**: The user fills out the Prediction Form (`predict.html`) and hits "Generate AI Prediction" or logs into the app.
2. **FastAPI Request processing**: The browser sends an HTTP POST request to the FastAPI backend running in its Docker container.
3. **CPU Spike**: FastAPI must unpack the payload, run the input through the complex 5-model Machine Learning Ensemble, run Database operations (SQLAlchemy), and format the response. This heavy mathematical operation causes an immediate spike in CPU usage for the Python process.
4. **Memory Allocation**: To run those models, RAM is actively allocated and held, causing a bump in memory usage.
5. **Prometheus Scraping**: Every 15 seconds, Prometheus reaches out to `node_exporter` (metrics of the host machine itself) and the FastAPI `/metrics` endpoint. It pulls the newly elevated CPU usage metrics, RAM usage metrics, and overall HTTP Request totals.
6. **Grafana Visualization**: Grafana continuously polls Prometheus for this time-series data. The line graphs instantly update to reflect the spike, proving the monitoring stack is capturing real-time app load dynamically.

### Grafana Queries to Try
To visualize the traffic and load on Grafana, go to **Explore**:
- **Total HTTP Requests**: `http_requests_total`
- **Request Rate (load per second)**: `rate(http_requests_total[5m])`
- **CPU Usage**: `rate(process_cpu_seconds_total[1m])`
- **Memory Consumption**: `process_resident_memory_bytes`

---

## 🔄 Verification & Page Logic Checks

1. **Home & About Pages (`/`, `/about`)**: Accessible entirely **without login**. Kept open by design so users can learn about the product before creating an account. The Navbar intelligently locks `Predict`, `Dashboard`, and `Logs` with lock icons when unauthenticated.
2. **Google OAuth**: Verified and functional. Handled correctly inside `main.py` routing using the Google Auth flow. Needs the redirect URI `http://localhost:8081/auth/google` whitelisted in the GCP Console.
3. **Scripts Folder**: Was successfully analyzed; contained an old `setup_local.py` file which unnecessarily overwrote `.env` files. **Safely Deleted.**
4. **Frontend Redesign**: Home, Login, Signup, OTP Verification, Complete Google-Signup, and Predict pages conform strictly to the premium `.agent/frontend-specialist` design principles (Dark Mode, Sage accents, Glassmorphism, Clean typography).

---

## 🔗 Related Documentation
- `Jenkinsfile` - Contains the full 6-stage pipeline (Build, Lint, Secure, Deploy).
- `docker-compose.yml` - Contains local development environment maps.
