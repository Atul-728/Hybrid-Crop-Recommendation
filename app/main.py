from fastapi import FastAPI, Request, Form, Depends
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from .database import engine, Base, SessionLocal
from .models import PredictionLog, User
from .prediction_service import predict_crop
from .visualization import generate_class_distribution
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
    return templates.TemplateResponse("login.html", {"request": request})

# ----------------------------
# REGISTER
# ----------------------------
@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/register", response_class=HTMLResponse)
async def register(
    request: Request,
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

    # Check for existing records
    existing_email = db.query(User).filter(User.email == email).first()
    existing_username = db.query(User).filter(User.username == username).first()

    if existing_email:
        if existing_email.is_verified:
            # Strictly One Email = One User
            return templates.TemplateResponse("register.html", {"request": request, "error": "This email is already registered. Please login."})
        else:
            # Overwrite unverified dummy user instead of blocking
            user_to_update = existing_email
    else:
        if existing_username:
            if existing_username.is_verified:
                return templates.TemplateResponse("register.html", {"request": request, "error": "Username is already taken."})
            else:
                # Clean up abandoned unverified username to free it up
                db.delete(existing_username)
                db.commit()
        
        # Create fresh dummy user
        user_to_update = User(email=email, role="farmer", is_verified=False)
        db.add(user_to_update)

    # Generate OTP
    otp = str(random.randint(100000, 999999))

    # Update credentials and OTP details
    user_to_update.username = username
    user_to_update.password = hash_password(password)
    user_to_update.otp_code = otp
    user_to_update.otp_expiry = datetime.utcnow() + timedelta(minutes=5)
    
    db.commit()

    # Send Email Asynchronously
    message = MessageSchema(
        subject="Your OTP Verification Code",
        recipients=[email],
        body=f"Your OTP is: {otp}\nValid for 5 minutes.",
        subtype="plain"
    )

    try:
        fm = FastMail(conf)
        await fm.send_message(message)
    except Exception as e:
        print("Email sending failed:", e)

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

    return RedirectResponse(url="/dashboard", status_code=303)

# ----------------------------
# DASHBOARD
# ----------------------------
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    generate_class_distribution()
    return templates.TemplateResponse("index.html", {"request": request})

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
    db: Session = Depends(get_db)
):
    input_data = {
        "N": N, "P": P, "K": K,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph, "rainfall": rainfall
    }

    result = predict_crop(input_data)

    log = PredictionLog(
        N=N, P=P, K=K,
        temperature=temperature,
        humidity=humidity,
        ph=ph, rainfall=rainfall,
        predicted_crop=result["top1"],
        confidence=result["confidence"]
    )

    db.add(log)
    db.commit()

    return templates.TemplateResponse("result.html", {"request": request, "result": result})

# ----------------------------
# VIEW & DOWNLOAD LOGS
# ----------------------------
@app.get("/logs", response_class=HTMLResponse)
def view_logs(request: Request, db: Session = Depends(get_db)):
    logs = db.query(PredictionLog).all()
    return templates.TemplateResponse("logs.html", {"request": request, "logs": logs})

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