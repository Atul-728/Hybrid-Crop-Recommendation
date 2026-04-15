# 🌱 CropOracle — Hybrid Crop AI Platform

Welcome to **CropOracle**, a fully complete, AI-powered agricultural platform designed to help farmers predict the best crops for their specific soil and climate conditions.

---

## 📖 What is CropOracle?

CropOracle acts as an advanced digital agricultural expert. 

Farmers or agricultural businesses can input their soil metrics (Nitrogen, Phosphorous, Potassium, pH levels) and regional climate details. CropOracle takes this information and processes it through a highly advanced Machine Learning brain (a "Stacked Hybrid Ensemble" combining 5 different prediction models) to confidently recommend the absolute best crop to grow. 

But it doesn't stop there. Once a crop is chosen, CropOracle immediately connects to our **AgentRouter AI (powered by DeepSeek-v3.1)** to:
1. Autocomplete the crops commonly grown in their specific city/state.
2. Fetch the **real-time market economics** (expected market price vs production cost per quintal) so the farmer knows their expected profit margin.
3. Provide a **24/7 AI Chatbot** embedded into the website to answer any sudden farming questions.

---

## 🛠️ The "Two Ways" to Run This Application

You can run CropOracle in two different ways depending on your goal. You can run it locally for simple testing, or you can run it using full "DevOps" tools just like mega-corporations (Google, Netflix, Amazon) do.

### 🏗️ Why Use DevOps Tools?

While running the project locally using simple terminal commands is perfect for development and testing, deploying CropOracle to the real world requires enterprise infrastructure:
*   **Docker:** If you send your code to a friend, it might crash because they don't have the right Python version. Docker creates an unbreakable "shipping container" that packages your code, Python, and the database together so it runs flawlessly on *any* computer, guaranteed.
*   **Kubernetes:** If 100,000 Indian farmers log into CropOracle at the same time, the server will melt. Kubernetes acts as a manager. If traffic is high, Kubernetes will automatically clone your app into 10 copies to handle the load. If one copy crashes, Kubernetes kills it and spawns a new one instantly ("self-healing").
*   **Prometheus & Grafana:** When thousands of people use the site, how do you know if the server is struggling? Prometheus constantly measures your server's "vitals" (CPU, Memory, Traffic), and Grafana turns those numbers into beautiful, colorful graphs. This allows administrators to fix problems *before* the website crashes.

---

## 🚀 Execution Commands (In Exact Sequence)

Here are the step-by-step terminal commands to run this project depending on your needs.

### Method A: The Simple Local Method (No DevOps Needed)
Use this if you just want to quickly test the website on your own laptop without launching heavy background infrastructure.

1. **Start the Application:**
   ```bash
   uvicorn Backend.main:app --host 127.0.0.1 --port 8001 --reload
   ```
2. **Access the Website:** Open your browser and go to `http://127.0.0.1:8001`

---

### Method B: The Production Method (Docker Compose)
Use this if you want to run the full Database and Application inside isolated Docker containers.

1. **Start Docker:**
   ```bash
   docker-compose up -d --build
   ```
2. **Access the Website:** Open your browser and go to `http://localhost:8081`

---

### Method C: The Enterprise Level (Kubernetes)
Use this to simulate a massive cloud-scale deployment using Minikube. Run these commands strictly in this order:

**⚠️ IMPORTANT PORT RULE:** You can only run ONE method at a time. Before starting Kubernetes, ensure that Method A (uvicorn) and Method B (docker-compose) are completely stopped to prevent port conflicts on `8080` and `8081`.

1. **Start the Mini-Cloud Cluster & Enable Monitor:**
   ```bash
   minikube start --driver=docker
   minikube addons enable metrics-server  # Required for Auto-scaling
   ```

2. **Build the Image Inside the Cluster (Critical Step):**
   Kubernetes needs the Docker image available locally. Point your terminal to Minikube's Docker Engine and build the image directly inside it:
   ```bash
   minikube docker-env | Invoke-Expression  # For Windows PowerShell
   # OR: eval $(minikube docker-env)        # For Mac/Linux

   docker build -t croporacle:latest .
   ```
   
3. **Push the Storage & Database configs:**
   ```bash
   kubectl apply -f k8s/postgres-pvc.yaml
   kubectl apply -f k8s/postgres-deployment.yaml
   ```

4. **Push the Application Configuration:**
   *(Note: The `k8s/configmap.yaml` file contains sensitive API keys needed for Kubernetes to run. For security, it is intentionally excluded from the public GitHub repository).*
   ```bash
   kubectl apply -f k8s/configmap.yaml
   kubectl apply -f k8s/deployment.yaml
   kubectl apply -f k8s/hpa.yaml         # Starts the Auto-scaler
   ```

5. **Verify it is Running:**
   *(Wait 1 minute, then check if the "Pods" are active)*
   ```bash
   kubectl get pods
   ```

6. **Expose It To Your Browser:**
   ```bash
   kubectl port-forward svc/croporacle-service 8080:8080
   ```
   *(Now go to `http://localhost:8080` in your browser)*

7. **How to Demonstrate Scaling (The Kubernetes Magic):**

   **Option A: Manual Scaling**
   If you want to show how Kubernetes handles high traffic manually, run this command:
   ```bash
   kubectl scale deployment croporacle-app --replicas=5
   ```
   
   **Option B: Automatic - **Auto-Scaling (HPA):** Scales from 1 to 5 replicas based on CPU load.
   - **Resource Baseline:** 500m CPU (0.5 Core) / 512Mi RAM.
   - **Scaling Threshold:** Triggers at 75% of baseline utilization.
    1. Open a terminal and run: `kubectl get hpa -w`
    2. Open the website and refresh the "Predict" page frequently to generate load.
    3. Within 1-2 minutes, if the CPU usage hits the **75% threshold**, you will see the "REPLICAS" column automatically jump from 1 to 5!

---

### Method D: The Monitoring System (Prometheus & Grafana)
Use this to launch the graphical dashboards that measure CPU and Memory loads. 

**Note:** The monitoring stack is independent and can be used to watch the application whether it is running via Docker Compose or Kubernetes.

1. **Start the Dashboards:**
   ```bash
   cd monitoring
   docker-compose -f docker-compose.monitoring.yml up -d
   ```

2. **Access the Monitoring Tools:**
   *   **Grafana (The Visualizer):** Visit `http://localhost:3000`.
       *   *Login: Username: `admin` | Password: `admin`*
       *   **Zero-Config Dashboard:** You don't need to setup anything! The "🌱 CropOracle - Multi-Node Monitoring" dashboard is pre-loaded as your home page.
   *   **Prometheus (The Data Collector):** Visit `http://localhost:9090`. 

3. **📊 Manual Metric Queries (PromQL):**
   The following queries can be used in the Prometheus or Grafana metrics browser for deep-dive technical analysis:
   
   | Metric | Query | Description |
   | :--- | :--- | :--- |
   | **CPU Load** | `rate(process_cpu_seconds_total[1m]) * 100` | Real-time CPU utilization % |
   | **Memory** | `process_resident_memory_bytes / 1024 / 1024` | Resident memory set in MB |
   | **Traffic** | `sum(rate(http_requests_total[1m]))` | Global requests per second |
   | **Errors** | `sum(rate(http_requests_total{status=~"5.."}[1m]))` | Internal Server Error tracking |

4. **🔭 Observability Insights:**
   This monitoring stack provides real-time visibility into the infrastructure. To observe system behavior under load, perform several crop predictions on the live website. Prometheus will capture the resulting CPU spike, which Grafana will visualize instantly. 
   
   **Architecture Note:** When CPU utilization exceeds the configured threshold (e.g., 75%), the Kubernetes Horizontal Pod Autoscaler (HPA) will dynamically spin up additional replicas (up to 5) to maintain platform stability. This can be verified via `kubectl get pods`.
