# Crop Recommendation System

## Overview

Crop Recommendation System is a machine learning based application that predicts the most suitable crop to grow based on soil nutrients and environmental conditions. The system analyzes parameters like nitrogen, phosphorus, potassium, temperature, humidity, pH, and rainfall to recommend the best crop.

The application is built using Python, FastAPI, and machine learning models for prediction.

---

## Features

• Machine learning based crop prediction
• FastAPI backend for prediction API
• Email notification support
• Data driven agricultural recommendations

---

## Technologies Used

• Python
• FastAPI
• Uvicorn
• CatBoost / Scikit-learn
• Pandas
• NumPy

---

## Recommended Setup

It is recommended to create a **virtual environment** before running the project.

### Create Virtual Environment

```
python -m venv venv
```

### Activate Virtual Environment

Windows

```
venv\Scripts\activate
```

Linux / Mac

```
source venv/bin/activate
```

---

## Install Dependencies

```
pip install -r requirements.txt
```

---

## Environment Variables

Create a file named **.env** in the root directory.

Example:

```
MAIL_USERNAME=username
EMAIL_PASSWORD=your_app_password
MAIL_FROM=your_email@gmail.com
```

These credentials are used for sending emails from the system.

---

## Generate Gmail App Password

Google requires an **App Password** instead of your Gmail password.

Steps:

1. Enable **2 Step Verification** in your Google account.
2. Open the App Password page.

Create it here:

https://myaccount.google.com/apppasswords

Select:
App → Mail
Device → Other
Name → Crop System

Google will generate a **16 character password**.
Use that password in the **.env file**.

---

## Running the Project

Run the project using Uvicorn.

### Run on specific port (recommended)

```
uvicorn app.main:app --port 8081 --reload
```

Use this when you want the server to run on **port 8081**.

### Run on default port

```
uvicorn app.main:app --reload
```

Use this when no specific port is required. The server will run on the **default port**.

---

## Author

Atul Kumar
B.Tech Computer Science Engineering
