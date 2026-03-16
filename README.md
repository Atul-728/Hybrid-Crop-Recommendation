Hybrid Crop Recommendation System
Overview

The Hybrid Crop Recommendation System is a machine learning based application designed to recommend the most suitable crop to cultivate based on soil nutrients and environmental conditions.

The system analyzes key agricultural parameters such as nitrogen (N), phosphorus (P), potassium (K), temperature, humidity, pH level, and rainfall to predict the optimal crop.

Unlike traditional single model systems, this project implements a hybrid ensemble architecture combining multiple machine learning models with uncertainty estimation and model explainability.

The application is built using FastAPI for the backend API and multiple machine learning models for prediction.

Key Features

• Hybrid ensemble model for crop prediction
• Multiple ML algorithms including gradient boosting models and deep learning
• TabNet model for deep learning based tabular data learning
• Stacked ensemble model for improved prediction accuracy
• Monte Carlo based uncertainty estimation for prediction confidence
• SHAP explainability for model interpretability
• FastAPI backend for scalable API deployment
• Email notification support for system outputs
• Data driven agricultural decision support

Machine Learning Architecture

The system uses a multi model hybrid architecture combining several models to improve prediction robustness.

Base models used:

• Random Forest
• XGBoost
• LightGBM
• CatBoost
• TabNet (Deep Learning for Tabular Data)

These models are combined using a stacked ensemble approach where predictions from base models are used by a meta learner to produce the final recommendation.

Additional research level components include:

• SHAP (SHapley Additive Explanations) for model interpretability
• Monte Carlo based uncertainty estimation to evaluate prediction confidence

This architecture allows the system to produce accurate, explainable, and reliable crop recommendations.

Technologies Used

Backend

• Python
• FastAPI
• Uvicorn

Machine Learning

• Scikit-learn
• XGBoost
• LightGBM
• CatBoost
• TabNet

Data Processing

• Pandas
• NumPy

Explainability

• SHAP

Utilities

• Python Dotenv
• ReportLab (PDF generation)

Project Structure
app/
    main.py
    predict_api.py
    price_fallback.py
    crop_mapping.py

ml/
    preprocess.py
    train_models.py
    ensemble.py
    explainability.py
    uncertainty.py
    tabnet_model.py

models/
    model_rf.pkl
    model_xgb.pkl
    model_lgbm.pkl
    model_catboost.pkl
    stacked_model.pkl

static/
templates/
data/
requirements.txt
Recommended Setup

It is recommended to create a virtual environment before running the project.

Create virtual environment

python -m venv venv

Activate virtual environment

Windows

venv\Scripts\activate

Linux / Mac

source venv/bin/activate
Install Dependencies
pip install -r requirements.txt
Environment Variables

Create a file named .env in the root directory.

Example:

MAIL_USERNAME=username
MAIL_PASSWORD=your_app_password
MAIL_FROM=your_email@gmail.com

These credentials are used for sending automated email notifications.

Generate Gmail App Password

Google requires an App Password instead of your Gmail password.

Steps

1 Enable 2 Step Verification in your Google account
2 Open the App Password page

Create it here

https://myaccount.google.com/apppasswords

Select

App → Mail
Device → Other
Name → Crop System

Google will generate a 16 character password.
Use that password in the .env file.

Running the Project

Run the FastAPI application using Uvicorn.

Run on specific port

uvicorn app.main:app --port 8081 --reload

Run on default port

uvicorn app.main:app --reload

After starting the server the application will be available at

http://127.0.0.1:8081
Future Improvements

• Integration with real time agricultural market price APIs
• Deployment using Docker and Kubernetes
• Integration with cloud infrastructure such as AWS
• Mobile application support for farmers

Author

Atul Kumar
B.Tech Computer Science Engineering