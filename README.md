# 🌱 CropOracle — Hybrid Crop AI Platform

Welcome to **CropOracle**, a fully complete, AI-powered agricultural platform designed to help farmers predict the best crops for their specific soil and climate conditions.

---

## 📖 What is CropOracle? (For Everyone)

If you are not from a technical background, think of CropOracle as a digital agricultural expert. 

Farmers or agricultural businesses can input their soil metrics (Nitrogen, Phosphorous, Potassium, pH levels) and regional climate details. CropOracle takes this information and processes it through a highly advanced Machine Learning brain (a "Stacked Hybrid Ensemble" combining 5 different prediction models) to confidently recommend the absolute best crop to grow. 

But it doesn't stop there. Once a crop is chosen, CropOracle immediately connects to our **AgentRouter AI (powered by DeepSeek-v3.1)** to:
1. Autocomplete the crops commonly grown in their specific city/state.
2. Fetch the **real-time market economics** (expected market price vs production cost per quintal) so the farmer knows their expected profit margin.
3. Provide a **24/7 AI Chatbot** embedded into the website to answer any sudden farming questions.

---

## 🛠️ The "Two Ways" to Run This Application

You can run CropOracle in two different ways depending on your goal. You can run it locally for simple testing, or you can run it using full "DevOps" tools just like mega-corporations (Google, Netflix, Amazon) do.

### Q: "If I can run it simply, why do we need DevOps tools like Docker and Kubernetes?"

That's a great question! Running the project locally (just typing `uvicorn` in the terminal) is perfect for a single student on a laptop. But if you were releasing CropOracle to the actual world:
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

1. **Start the Mini-Cloud Cluster:**
   ```bash
   minikube start --driver=docker
   ```
   
2. **Push the Storage & Database configs:**
   ```bash
   kubectl apply -f k8s/postgres-pvc.yaml
   kubectl apply -f k8s/postgres-deployment.yaml
   ```

3. **Push the Application Configuration:**
   ```bash
   kubectl apply -f k8s/configmap.yaml
   kubectl apply -f k8s/deployment.yaml
   ```

4. **Verify it is Running:**
   *(Wait 1 minute, then check if the "Pods" are active)*
   ```bash
   kubectl get pods
   ```

5. **Expose It To Your Browser:**
   ```bash
   kubectl port-forward svc/croporacle-service 8080:8080
   ```
   *(Now go to `http://localhost:8080` in your browser)*

---

### Method D: The Monitoring System (Prometheus & Grafana)
Use this to launch the graphical dashboards that measure CPU and Memory loads.

1. **Start the Dashboards:**
   ```bash
   cd monitoring
   docker-compose -f docker-compose.monitoring.yml up -d
   ```
2. **View the Graphs:** Open your browser and go to `http://localhost:3000`
3. **Login Details:** Username: `admin`, Password: `admin`
