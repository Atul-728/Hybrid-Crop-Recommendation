from fastapi import FastAPI, Request, Form, Depends, Cookie
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from .database import engine, Base, SessionLocal
from .models import PredictionLog, User
from .prediction_service import predict_crop
from .auth import hash_password, verify_password
from fastapi.staticfiles import StaticFiles
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
import os
import csv
import secrets
import random
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

Base.metadata.create_all(bind=engine)
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# ----------------------------
# MAIL CONFIG
# ----------------------------
BLOCKED_DOMAINS = [
    "tempmail.com",
    "10minutemail.com",
    "mailinator.com",
    "guerrillamail.com"
]

conf = ConnectionConfig(
    MAIL_USERNAME=os.getenv("MAIL_USERNAME"),
    MAIL_PASSWORD=os.getenv("MAIL_PASSWORD"),
    MAIL_FROM=os.getenv("MAIL_FROM"),
    MAIL_PORT=587,
    MAIL_SERVER="smtp.gmail.com",
    MAIL_STARTTLS=True,
    MAIL_SSL_TLS=False,
    USE_CREDENTIALS=True
)

# ----------------------------
# Database Dependency
# ----------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ----------------------------
# HOME (LOGIN FIRST)
# ----------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    if request.cookies.get("user_id"):
        return RedirectResponse(url="/dashboard", status_code=303)
    return templates.TemplateResponse("login.html", {"request": request})

# ----------------------------
# REGISTER
# ----------------------------
@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request):
    if request.cookies.get("user_id"):
        return RedirectResponse(url="/dashboard", status_code=303)
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/register", response_class=HTMLResponse)
async def register(
    request: Request,
    first_name: str = Form(...),
    last_name: str = Form(...),
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    if "@" not in email:
        return templates.TemplateResponse("register.html", {"request": request, "error": "Invalid email format."})

    domain = email.split("@")[1]
    if domain in BLOCKED_DOMAINS:
        return templates.TemplateResponse("register.html", {"request": request, "error": "Temporary email addresses are not allowed."})

    existing_email = db.query(User).filter(User.email == email).first()
    existing_username = db.query(User).filter(User.username == username).first()

    if existing_email and existing_email.is_verified:
        return templates.TemplateResponse("register.html", {"request": request, "error": "Email already registered."})
    
    if existing_username and existing_username.is_verified:
        return templates.TemplateResponse("register.html", {"request": request, "error": "Username taken."})

    user = existing_email if existing_email else User(email=email, role="farmer", is_verified=False)
    user.first_name = first_name.strip()
    user.last_name = last_name.strip()
    user.username = username
    user.password = hash_password(password)
    
    otp = str(random.randint(100000, 999999))
    user.otp_code = otp
    user.otp_expiry = datetime.utcnow() + timedelta(minutes=5)
    
    if not existing_email:
        db.add(user)
    db.commit()

    message = MessageSchema(
        subject="Your OTP Verification Code",
        recipients=[email],
        body=f"Your OTP is: {otp}",
        subtype="plain"
    )
    try:
        fm = FastMail(conf)
        await fm.send_message(message)
    except Exception as e:
        print("Email failed:", e)

    return templates.TemplateResponse("otp_verify.html", {"request": request, "email": email})

# ----------------------------
# VERIFY OTP
# ----------------------------
@app.post("/verify-otp", response_class=HTMLResponse)
def verify_otp(
    request: Request,
    email: str = Form(...),
    otp: str = Form(...),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.email == email).first()

    if not user:
        return templates.TemplateResponse("register.html", {"request": request, "error": "User session lost. Please register again."})
    remaining_time = int((user.otp_expiry - datetime.utcnow()).total_seconds())
    time_left = max(0, remaining_time)

    if user.otp_code != otp:
        return templates.TemplateResponse("otp_verify.html", {"request": request, "email": email, "error": "Invalid OTP. Please try again.", "time_left": time_left})

    if datetime.utcnow() > user.otp_expiry:
        return templates.TemplateResponse("otp_verify.html", {"request": request, "email": email, "error": "OTP has expired. Please click Resend OTP.", "is_expired": True, "time_left": 0})

    # Success: Mark as verified and clear OTP
    user.is_verified = True
    user.otp_code = None
    user.otp_expiry = None
    db.commit()

    return templates.TemplateResponse("login.html", {"request": request, "success": "Account verified successfully. Please login."})

# ----------------------------
# RESEND OTP
# ----------------------------
@app.post("/resend-otp", response_class=HTMLResponse)
async def resend_otp(
    request: Request,
    email: str = Form(...),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.email == email).first()

    if not user:
        return templates.TemplateResponse("register.html", {"request": request, "error": "User session lost. Please register again."})

    if user.is_verified:
        return templates.TemplateResponse("login.html", {"request": request, "error": "Account is already verified. Please login."})

    # Generate new OTP and extend expiry by 5 minutes
    new_otp = str(random.randint(100000, 999999))
    user.otp_code = new_otp
    user.otp_expiry = datetime.utcnow() + timedelta(seconds=30)
    db.commit()

    message = MessageSchema(
        subject="Your New OTP Verification Code",
        recipients=[email],
        body=f"Your new OTP is: {new_otp}\nValid for 5 minutes.",
        subtype="plain"
    )

    try:
        fm = FastMail(conf)
        await fm.send_message(message)
    except Exception as e:
        print("Email sending failed:", e)

    return templates.TemplateResponse("otp_verify.html", {"request": request, "email": email,"success": "A new OTP has been sent to your email.", "time_left": 30})

# ----------------------------
# LIVE USERNAME & EMAIL CHECK
# ----------------------------
@app.get("/check-username")
def check_username(username: str, db: Session = Depends(get_db)):
    # Only consider verified users as "taken" to prevent unverified trolls from locking names
    user = db.query(User).filter(User.username == username, User.is_verified == True).first()
    if user:
        return JSONResponse({"available": False})
    return JSONResponse({"available": True})

@app.get("/check-email")
def check_email(email: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == email, User.is_verified == True).first()
    if user:
        return JSONResponse({"available": False})
    return JSONResponse({"available": True})

# ----------------------------
# LOGIN
# ----------------------------
@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    if request.cookies.get("user_id"):
        return RedirectResponse(url="/dashboard", status_code=303)
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter((User.username == username) | (User.email == username)).first()

    if not user or not verify_password(password, user.password):
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials."})

    if not user.is_verified:
        return templates.TemplateResponse("login.html", {"request": request, "error": "Please verify your email before logging in."})
    
    response = RedirectResponse(url="/dashboard", status_code=303)
    response.set_cookie(key="user_id", value=str(user.id), httponly=True)
    return response

@app.get("/logout")
def logout():
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie("user_id") # Cookie delete kar di
    return response

# ----------------------------
# DASHBOARD
# ----------------------------
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request, db: Session = Depends(get_db)):
    user_id = request.cookies.get("user_id")
    if not user_id:
        return RedirectResponse(url="/login", status_code=303)
    
    user = db.query(User).filter(User.id == int(user_id)).first()
    
    # Logic: Agar koi log nahi hai to "Welcome", warna "Welcome Back"
    has_logs = db.query(PredictionLog).filter(PredictionLog.user_id == int(user_id)).first()
    prefix = "Welcome to the family," if not has_logs else "Welcome back,"
    
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "name": f"{user.first_name} {user.last_name}",
        "greeting_prefix": prefix
    })

# ----------------------------
# FORGOT PASSWORD FLOW
# ----------------------------
@app.get("/forgot-password", response_class=HTMLResponse)
def forgot_password_page(request: Request):
    return templates.TemplateResponse("forgot_password.html", {"request": request})

@app.post("/send-reset-otp")
async def send_reset_otp(email: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == email).first()

    if not user:
        return JSONResponse({"success": False, "message": "This email is not registered with us."})
    if not user.is_verified:
        return JSONResponse({"success": False, "message": "This account is not verified yet."})

    # Set OTP expiry to 40 seconds for testing (Change to minutes=5 for production)
    otp = str(random.randint(100000, 999999))
    user.otp_code = otp
    user.otp_expiry = datetime.utcnow() + timedelta(minutes=5)
    db.commit()

    message = MessageSchema(
        subject="Password Reset OTP",
        recipients=[email],
        body=f"Your OTP for password reset is: {otp}\nValid for testing.",
        subtype="plain"
    )

    try:
        fm = FastMail(conf)
        await fm.send_message(message)
        return JSONResponse({"success": True})
    except Exception as e:
        print("Email sending failed:", e)
        return JSONResponse({"success": False, "message": "Failed to send email. Check connection."})

@app.post("/verify-reset-otp")
def verify_reset_otp(email: str = Form(...), otp: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == email).first()

    if not user or user.otp_code != otp:
        return JSONResponse({"success": False, "message": "Invalid OTP. Please try again."})

    if datetime.utcnow() > user.otp_expiry:
        return JSONResponse({"success": False, "message": "OTP has expired. Please request a new one."})

    return JSONResponse({"success": True})

@app.post("/reset-password")
def reset_password(
    email: str = Form(...), 
    otp: str = Form(...), 
    new_password: str = Form(...), 
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.email == email).first()

    if not user or user.otp_code != otp:
        return JSONResponse({"success": False, "message": "Invalid OTP."})

    if datetime.utcnow() > user.otp_expiry:
        return JSONResponse({"success": False, "message": "OTP has expired. Please request a new one."})

    user.password = hash_password(new_password)
    user.otp_code = None
    user.otp_expiry = None
    db.commit()

    return JSONResponse({"success": True})

# ----------------------------
# PREDICTION
# ----------------------------
@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    N: float = Form(...),
    P: float = Form(...),
    K: float = Form(...),
    temperature: float = Form(...),
    humidity: float = Form(...),
    ph: float = Form(...),
    rainfall: float = Form(...),
    region: str = Form(...),
    state: str = Form(...),
    user_id: str = Cookie(None),
    db: Session = Depends(get_db)
):
    if not user_id:
        return RedirectResponse(url="/login", status_code=303)

    input_data = {
        "N": N,
        "P": P,
        "K": K,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall,
        "region": region,
        "state": state,
        "market_price": 0,
        "production_cost": 0
    }
    result = predict_crop(input_data)
    new_log = PredictionLog(
        N=N, P=P, K=K,
        temperature=temperature,
        humidity=humidity,
        ph=ph,
        rainfall=rainfall,
        region=region,
        state=state,
        market_price=result.get("market_price", 0),
        production_cost=result.get("production_cost", 0),
        predicted_crop=result["top1"],
        confidence=result["confidence"],
        risk_score=result["risk_score"],
        expected_profit=result["expected_profit"],
        user_id=int(user_id)
    )
    db.add(new_log)
    db.commit()

    user = db.query(User).filter(User.id == int(user_id)).first()
    return templates.TemplateResponse("result.html", {"request": request, "result": result})

# ----------------------------
# VIEW & DOWNLOAD LOGS
# ----------------------------
@app.get("/logs", response_class=HTMLResponse)
def view_logs(request: Request, user_id: str = Cookie(None), db: Session = Depends(get_db)):
    if not user_id:
        return RedirectResponse(url="/login", status_code=303)   
    user = db.query(User).filter(User.id == int(user_id)).first()
    logs = db.query(PredictionLog).filter(PredictionLog.user_id == int(user_id)).all()
    return templates.TemplateResponse("logs.html", {"request": request, "logs": logs, "username": user.username})
# ----------------------------
# ML ANALYTICS DASHBOARD
# ----------------------------
@app.get("/analytics", response_class=HTMLResponse)
def analytics_page(request: Request, user_id: str = Cookie(None), db: Session = Depends(get_db)): # db add kiya
    if not user_id:
        return RedirectResponse(url="/login", status_code=303) 
    user = db.query(User).filter(User.id == int(user_id)).first()
    return templates.TemplateResponse("analytics.html", {"request": request, "username": user.username})
# ----------------------------
# VIEW SINGLE LOG (Re-run prediction to show result)
# ----------------------------
@app.get("/view-log/{log_id}", response_class=HTMLResponse)
def view_single_log(request: Request, log_id: int, db: Session = Depends(get_db)):
    user_id = request.cookies.get("user_id")
    if not user_id:
        return RedirectResponse(url="/login", status_code=303)

    log = db.query(PredictionLog).filter(PredictionLog.id == log_id, PredictionLog.user_id == int(user_id)).first()
    if not log:
        return RedirectResponse(url="/logs", status_code=303)

    # Re-run the exact same prediction using saved data
    input_data = {
        "N": log.N, "P": log.P, "K": log.K,
        "temperature": log.temperature, "humidity": log.humidity,
        "ph": log.ph, "rainfall": log.rainfall,
        "region": log.region or "North",
        "state": log.state or "",
        "market_price": log.market_price or 0, 
        "production_cost": log.production_cost or 0
    }
    result = predict_crop(input_data)

    user = db.query(User).filter(User.id == int(user_id)).first()
    return templates.TemplateResponse("result.html", {"request": request, "result": result})

# ----------------------------
# DELETE SINGLE LOG
# ----------------------------
@app.post("/delete-log/{log_id}")
def delete_log(log_id: int, request: Request, db: Session = Depends(get_db)):
    user_id = request.cookies.get("user_id")
    if user_id:
        log = db.query(PredictionLog).filter(PredictionLog.id == log_id, PredictionLog.user_id == int(user_id)).first()
        if log:
            db.delete(log)
            db.commit()
    return RedirectResponse(url="/logs", status_code=303)

# ----------------------------
# CLEAR ALL LOGS
# ----------------------------
@app.post("/clear-logs")
def clear_logs(request: Request, db: Session = Depends(get_db)):
    user_id = request.cookies.get("user_id")
    if user_id:
        db.query(PredictionLog).filter(PredictionLog.user_id == int(user_id)).delete()
        db.commit()
    return RedirectResponse(url="/logs", status_code=303)


@app.get("/download_csv")
def download_csv(db: Session = Depends(get_db)):
    logs = db.query(PredictionLog).all()
    file_path = "prediction_logs.csv"

    with open(file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["N", "P", "K", "Temperature", "Humidity", "pH", "Rainfall", "Crop", "Confidence"])
        for log in logs:
            writer.writerow([
                log.N, log.P, log.K,
                log.temperature, log.humidity,
                log.ph, log.rainfall,
                log.predicted_crop, log.confidence
            ])

    return FileResponse(file_path, media_type='text/csv', filename="prediction_logs.csv")

@app.get("/download_pdf")
def download_pdf(db: Session = Depends(get_db)):
    logs = db.query(PredictionLog).all()
    file_path = "prediction_logs.pdf"
    
    doc = SimpleDocTemplate(file_path)
    elements = []
    styles = getSampleStyleSheet()

    for log in logs:
        text = f"{log.timestamp} - Crop: {log.predicted_crop} - Confidence: {log.confidence}%"
        elements.append(Paragraph(text, styles["Normal"]))
        elements.append(Spacer(1, 12))

    doc.build(elements)
    return FileResponse(file_path, media_type='application/pdf', filename="prediction_logs.pdf")