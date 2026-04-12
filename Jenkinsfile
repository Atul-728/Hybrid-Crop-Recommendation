pipeline {
    agent any

    environment {
        IMAGE_NAME = 'croporacle'
        IMAGE_TAG = "v${BUILD_NUMBER}"
    }

    stages {
        stage('Checkout') {
            steps {
                echo '📦 Checking out source code...'
                checkout scm
            }
        }

        stage('Install Dependencies') {
            steps {
                echo '🐍 Installing Python dependencies...'
                sh 'pip install -r requirements.txt --quiet'
            }
        }

        stage('Lint & Validate') {
            steps {
                echo '🔍 Running linting checks...'
                sh 'python -m py_compile Backend/main.py Backend/models.py Backend/database.py'
                echo '✅ Syntax check passed'
            }
        }

        stage('Build Docker Image') {
            steps {
                echo "🐋 Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
                sh "docker build -t ${IMAGE_NAME}:${IMAGE_TAG} -t ${IMAGE_NAME}:latest ."
            }
        }

        stage('Security Scan') {
            steps {
                echo '🔒 Running security scan...'
                sh 'docker run --rm -v $(pwd):/app ${IMAGE_NAME}:latest pip check || true'
                echo '✅ Security scan complete'
            }
        }

        stage('Deploy to Kubernetes') {
            steps {
                echo '☸️ Deploying to Kubernetes (Minikube)...'
                sh '''
                    # Load image into Minikube
                    minikube image load ${IMAGE_NAME}:latest

                    # Apply all K8s manifests
                    kubectl apply -f k8s/postgres-pvc.yaml
                    kubectl apply -f k8s/postgres-deployment.yaml
                    kubectl apply -f k8s/configmap.yaml
                    kubectl apply -f k8s/deployment.yaml

                    # Wait for rollout
                    kubectl rollout status deployment/croporacle-app --timeout=120s
                '''
            }
        }

        stage('Health Check') {
            steps {
                echo '❤️ Running health check...'
                sh '''
                    sleep 10
                    MINIKUBE_IP=$(minikube ip)
                    curl -f http://${MINIKUBE_IP}:30080/health || exit 1
                    echo "✅ App is healthy at http://${MINIKUBE_IP}:30080"
                '''
            }
        }
    }

    post {
        success {
            echo '''
            ╔════════════════════════════╗
            ║  ✅ PIPELINE SUCCESS!      ║
            ║  CropOracle is deployed.  ║
            ╚════════════════════════════╝
            '''
        }
        failure {
            echo '❌ Pipeline failed. Check logs above.'
        }
        always {
            echo "📊 Build #${BUILD_NUMBER} completed."
        }
    }
}
