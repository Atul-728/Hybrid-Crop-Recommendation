from fastapi import FastAPI, Request, Form, Depends, Cookie, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth
from .database import engine, Base, SessionLocal
from .models import PredictionLog, User
from .prediction_service import predict_crop, gemini_generate
from .auth import hash_password, verify_password
from fastapi.staticfiles import StaticFiles
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import os, csv, secrets, random, requests, json
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv
import google.generativeai as genai
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    PROMETHEUS_ENABLED = True
except ImportError:
    PROMETHEUS_ENABLED = False

load_dotenv()  # Load base config (.env) — Railway production values

# Load .env.local on top if it exists (local development overrides)
_local_env = Path(".env.local")
if _local_env.exists():
    load_dotenv(dotenv_path=_local_env, override=True)
    print("[ENV] Loaded .env.local overrides for local development")

Base.metadata.create_all(bind=engine)
app = FastAPI(title="CropOracle", description="Uncertainty-Aware Hybrid Crop Recommendation System")

# Required for Google Authlib state management
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SECRET_KEY", secrets.token_hex(32)))

app.mount("/static", StaticFiles(directory="Frontend/static"), name="static")
templates = Jinja2Templates(directory="Frontend")

# ----------------------------
# PROMETHEUS METRICS
# ----------------------------
if PROMETHEUS_ENABLED:
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# ----------------------------
# HEALTH CHECK
# ----------------------------
@app.get("/health")
def health_check():
    return {"status": "healthy", "app": "CropOracle", "version": "2.0.0"}

BLOCKED_DOMAINS = ["tempmail.com", "10minutemail.com", "mailinator.com", "guerrillamail.com"]
TEMP_USERS = {}

# ----------------------------
# GOOGLE OAUTH CONFIGURATION
# ----------------------------
oauth = OAuth()
if os.getenv("GOOGLE_CLIENT_ID"):
    oauth.register(
        name='google',
        client_id=os.getenv("GOOGLE_CLIENT_ID"),
        client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
        server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
        client_kwargs={'scope': 'openid email profile'},
    )

# ----------------------------
# EMAIL API HELPER (BREVO API)
# ----------------------------
def send_email_via_api(to_email, subject, body):
    api_key = os.getenv("BREVO_API_KEY")
    sender_email = os.getenv("MAIL_FROM")

    if not api_key or not sender_email:
        return

    url = "https://api.brevo.com/v3/smtp/email"
    headers = {
        "accept": "application/json",
        "api-key": api_key,
        "content-type": "application/json"
    }

    data = {
        "sender": {"name": "CropOracle", "email": sender_email},
        "to": [{"email": to_email}],
        "subject": subject,
        "textContent": body
    }

    try:
        requests.post(url, headers=headers, json=data, timeout=10)
    except Exception:
        pass

def validate_username(username: str) -> tuple[bool, str]:
    if not username or len(username) < 3:
        return False, "Username must be at least 3 characters long."
    if not username.isalnum():
        return False, "Username cannot contain spaces or special characters."
    if username.isdigit():
        return False, "Username cannot contain only numbers."
    return True, ""

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
# GOOGLE AUTH ROUTES
# ----------------------------
@app.get("/login/google")
async def login_google(request: Request, flow: str = "login"):
    if not os.getenv("GOOGLE_CLIENT_ID"):
        return RedirectResponse(url="/login?error=Google Login not configured.")
    request.session['auth_flow'] = flow
    base_url = os.getenv("APP_BASE_URL", "").rstrip("/")
    if base_url:
        redirect_uri = f"{base_url}/auth/google"
    else:
        redirect_uri = str(request.url_for('auth_google'))
        if "http://" in redirect_uri and "railway.app" in redirect_uri:
            redirect_uri = redirect_uri.replace("http://", "https://")
    return await oauth.google.authorize_redirect(request, redirect_uri)

@app.get("/auth/google")
async def auth_google(request: Request, db: Session = Depends(get_db)):
    try:
        token = await oauth.google.authorize_access_token(request)
        user_info = token.get('userinfo')
        email = user_info['email']
        flow = request.session.get('auth_flow', 'login')

        user = db.query(User).filter(User.email == email).first()

        if flow == "login":
            if not user:
                return RedirectResponse(url="/register?error=No account found with this Google email. Please sign up first.", status_code=303)
            if not user.is_verified:
                return RedirectResponse(url="/login?error=Account not verified. Please complete signup.", status_code=303)
            response = RedirectResponse(url="/dashboard", status_code=303)
            response.set_cookie(key="user_id", value=str(user.id), httponly=True)
            return response

        elif flow == "register":
            if user:
                return RedirectResponse(url="/login?error=Account already exists with this Google email. Please login.", status_code=303)

            TEMP_USERS[email] = {
                "first_name": user_info.get('given_name', ''),
                "last_name": user_info.get('family_name', ''),
                "email": email,
                "is_google": True
            }
            return RedirectResponse(url=f"/complete-google-signup?email={email}", status_code=303)

    except Exception as e:
        print(f"[Google OAuth Error] {e}")
        return RedirectResponse(url="/login?error=Google authentication failed. Please try again.")

@app.get("/complete-google-signup", response_class=HTMLResponse)
def complete_google_signup_page(request: Request, email: str):
    if email not in TEMP_USERS or not TEMP_USERS[email].get("is_google"):
        return RedirectResponse(url="/register?error=Invalid session. Please sign up again.")
    return templates.TemplateResponse("complete_signup.html", {"request": request, "email": email})

@app.post("/complete-google-signup", response_class=HTMLResponse)
def complete_google_signup(
    request: Request, background_tasks: BackgroundTasks,
    username: str = Form(...), email: str = Form(...),
    db: Session = Depends(get_db)
):
    temp_user = TEMP_USERS.get(email)
    if not temp_user or not temp_user.get("is_google"):
        return RedirectResponse(url="/register?error=Session expired. Please sign up again.")

    is_valid, error_msg = validate_username(username)
    if not is_valid:
        return templates.TemplateResponse("complete_signup.html", {"request": request, "email": email, "error": error_msg})

    if db.query(User).filter(User.username == username).first():
        return templates.TemplateResponse("complete_signup.html", {"request": request, "email": email, "error": "Username is already taken. Please choose another."})

    otp = str(random.randint(100000, 999999))
    TEMP_USERS[email]["username"] = username
    TEMP_USERS[email]["password"] = hash_password(secrets.token_hex(16))
    TEMP_USERS[email]["otp"] = otp
    TEMP_USERS[email]["expiry"] = datetime.utcnow() + timedelta(minutes=5)

    print(f"\n==========\n[GOOGLE SIGNUP] OTP FOR {email}: {otp}\n==========\n")
    background_tasks.add_task(
        send_email_via_api, email,
        "CropOracle — Verify Your Account",
        f"Welcome to CropOracle!\n\nYour verification OTP is: {otp}\nThis code is valid for 5 minutes.\n\nIf you didn't request this, ignore this email."
    )

    return templates.TemplateResponse("otp_verify.html", {"request": request, "email": email, "time_left": 300})

# ----------------------------
# HOME & REGISTER
# ----------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    user_id = request.cookies.get("user_id")
    return templates.TemplateResponse("home.html", {"request": request, "is_logged_in": bool(user_id)})

@app.get("/about", response_class=HTMLResponse)
def about_page(request: Request):
    user_id = request.cookies.get("user_id")
    return templates.TemplateResponse("about.html", {"request": request, "is_logged_in": bool(user_id)})

@app.get("/predict", response_class=HTMLResponse)
def predict_page(request: Request, db: Session = Depends(get_db)):
    user_id = request.cookies.get("user_id")
    if not user_id:
        return RedirectResponse(url="/login?next=/predict", status_code=303)
    user = db.query(User).filter(User.id == int(user_id)).first()
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse("predict.html", {
        "request": request,
        "name": f"{user.first_name} {user.last_name}",
        "username": user.username,
        "is_logged_in": True
    })

@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request):
    if request.cookies.get("user_id"):
        return RedirectResponse(url="/dashboard", status_code=303)
    error = request.query_params.get("error")
    return templates.TemplateResponse("register.html", {"request": request, "error": error})

@app.post("/register", response_class=HTMLResponse)
async def register(
    request: Request, background_tasks: BackgroundTasks,
    first_name: str = Form(...), last_name: str = Form(...), username: str = Form(...),
    email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)
):
    if "@" not in email:
        return templates.TemplateResponse("register.html", {"request": request, "error": "Invalid email format."})
    domain = email.split("@")[1]
    if domain in BLOCKED_DOMAINS:
        return templates.TemplateResponse("register.html", {"request": request, "error": "Temporary email addresses are not allowed."})

    is_valid, error_msg = validate_username(username)
    if not is_valid:
        return templates.TemplateResponse("register.html", {"request": request, "error": error_msg})

    if db.query(User).filter(User.email == email).first():
        return templates.TemplateResponse("register.html", {"request": request, "error": "Email already registered. Please login."})
    if db.query(User).filter(User.username == username).first():
        return templates.TemplateResponse("register.html", {"request": request, "error": "Username already taken. Please choose another."})

    otp = str(random.randint(100000, 999999))
    TEMP_USERS[email] = {
        "first_name": first_name.strip(), "last_name": last_name.strip(),
        "username": username, "password": hash_password(password),
        "otp": otp, "expiry": datetime.utcnow() + timedelta(minutes=5)
    }
    print(f"\n==========\n[FAILSAFE] OTP FOR {email}: {otp}\n==========\n")
    background_tasks.add_task(
        send_email_via_api, email,
        "CropOracle — Verify Your Account",
        f"Welcome to CropOracle!\n\nYour verification OTP is: {otp}\nThis code is valid for 5 minutes.\n\nIf you didn't request this, ignore this email."
    )
    return templates.TemplateResponse("otp_verify.html", {"request": request, "email": email, "time_left": 300})

@app.post("/verify-otp", response_class=HTMLResponse)
def verify_otp(request: Request, email: str = Form(...), otp: str = Form(...), db: Session = Depends(get_db)):
    temp_user = TEMP_USERS.get(email)
    if not temp_user:
        return templates.TemplateResponse("register.html", {"request": request, "error": "Session expired. Please register again."})

    # Check expiry FIRST (security: prevent timing attacks)
    remaining_time = int((temp_user["expiry"] - datetime.utcnow()).total_seconds())
    time_left = max(0, remaining_time)

    if datetime.utcnow() > temp_user["expiry"]:
        return templates.TemplateResponse("otp_verify.html", {
            "request": request, "email": email,
            "error": "OTP has expired. Please request a new one.",
            "is_expired": True, "time_left": 0
        })

    if temp_user["otp"] != otp:
        return templates.TemplateResponse("otp_verify.html", {
            "request": request, "email": email,
            "error": "Invalid OTP. Please try again.", "time_left": time_left
        })

    # OTP valid — create the user with proper error handling
    try:
        new_user = User(
            email=email,
            first_name=temp_user["first_name"],
            last_name=temp_user["last_name"],
            username=temp_user["username"],
            password=temp_user["password"],
            role="farmer",
            is_verified=True
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        # Only delete from temp store AFTER successful DB commit
        del TEMP_USERS[email]
        return templates.TemplateResponse("login.html", {
            "request": request,
            "success": "Account verified successfully! Please sign in."
        })
    except Exception as e:
        db.rollback()
        print(f"[OTP Verify DB Error] {e}")
        # Do not delete TEMP_USERS entry so user can retry
        return templates.TemplateResponse("otp_verify.html", {
            "request": request, "email": email,
            "error": "Account creation failed. This email may already be registered. Please try logging in.",
            "time_left": time_left
        })

@app.post("/resend-otp", response_class=HTMLResponse)
async def resend_otp(
    request: Request, background_tasks: BackgroundTasks,
    email: str = Form(...), db: Session = Depends(get_db)
):
    temp_user = TEMP_USERS.get(email)
    if not temp_user:
        if db.query(User).filter(User.email == email).first():
            return templates.TemplateResponse("login.html", {"request": request, "success": "Account already verified. Please login."})
        return templates.TemplateResponse("register.html", {"request": request, "error": "Session expired. Please register again."})

    new_otp = str(random.randint(100000, 999999))
    temp_user["otp"] = new_otp
    temp_user["expiry"] = datetime.utcnow() + timedelta(minutes=5)
    print(f"\n========== NEW OTP FOR {email} IS: {new_otp} ==========\n")
    background_tasks.add_task(
        send_email_via_api, email,
        "CropOracle — New Verification OTP",
        f"Your new CropOracle verification OTP is: {new_otp}\nValid for 5 minutes."
    )
    return templates.TemplateResponse("otp_verify.html", {
        "request": request, "email": email,
        "success": "New OTP sent to your email.", "time_left": 300
    })

# Live availability checks
@app.get("/check-username")
def check_username(username: str, db: Session = Depends(get_db)):
    return JSONResponse({"available": not db.query(User).filter(User.username == username).first()})

@app.get("/check-email")
def check_email(email: str, db: Session = Depends(get_db)):
    return JSONResponse({"available": not db.query(User).filter(User.email == email).first()})

# ----------------------------
# LOGIN & LOGOUT
# ----------------------------
@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    if request.cookies.get("user_id"):
        return RedirectResponse(url="/dashboard", status_code=303)
    return templates.TemplateResponse("login.html", {
        "request": request,
        "error": request.query_params.get("error"),
        "success": request.query_params.get("success")
    })

@app.post("/login")
def login(
    request: Request,
    username: str = Form(...), password: str = Form(...),
    db: Session = Depends(get_db)
):
    # Accept either username or email
    user = db.query(User).filter(
        (User.username == username) | (User.email == username)
    ).first()

    if not user or not user.password:
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials. Please check your username/email and password."})

    if not verify_password(password, user.password):
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials. Please check your username/email and password."})

    if not user.is_verified:
        return templates.TemplateResponse("login.html", {"request": request, "error": "Account not verified. Please complete email verification."})

    response = RedirectResponse(url="/dashboard", status_code=303)
    response.set_cookie(key="user_id", value=str(user.id), httponly=True)
    return response

@app.get("/logout")
def logout():
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie("user_id")
    return response

# ----------------------------
# FORGOT PASSWORD
# ----------------------------
@app.get("/forgot-password", response_class=HTMLResponse)
def forgot_password_page(request: Request):
    return templates.TemplateResponse("forgot_password.html", {"request": request})

@app.post("/send-reset-otp")
async def send_reset_otp(
    background_tasks: BackgroundTasks,
    email: str = Form(...),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.email == email).first()
    if not user:
        return JSONResponse({"success": False, "message": "No account found with this email address."})

    otp = str(random.randint(100000, 999999))
    user.otp_code = otp
    user.otp_expiry = datetime.utcnow() + timedelta(minutes=5)
    db.commit()
    print(f"\n========== RESET OTP FOR {email} IS: {otp} ==========\n")
    background_tasks.add_task(
        send_email_via_api, email,
        "CropOracle — Password Reset OTP",
        f"Your CropOracle password reset OTP is: {otp}\nValid for 5 minutes.\n\nIf you didn't request this, ignore this email."
    )
    return JSONResponse({"success": True})

@app.post("/verify-reset-otp")
def verify_reset_otp(email: str = Form(...), otp: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == email).first()
    if not user:
        return JSONResponse({"success": False, "message": "Email not found."})
    if datetime.utcnow() > user.otp_expiry:
        return JSONResponse({"success": False, "message": "OTP has expired. Please request a new one."})
    if user.otp_code != otp:
        return JSONResponse({"success": False, "message": "Invalid OTP."})
    return JSONResponse({"success": True})

@app.post("/reset-password")
def reset_password(
    email: str = Form(...), otp: str = Form(...),
    new_password: str = Form(...), db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.email == email).first()
    if not user or user.otp_code != otp:
        return JSONResponse({"success": False, "message": "Invalid OTP."})
    if datetime.utcnow() > user.otp_expiry:
        return JSONResponse({"success": False, "message": "OTP expired."})
    user.password = hash_password(new_password)
    user.otp_code = None
    user.otp_expiry = None
    db.commit()
    return JSONResponse({"success": True, "message": "Password updated successfully."})

# ----------------------------
# AI ENDPOINTS (Chatbot & Auto-Suggest)
# ----------------------------
@app.post("/api/chat")
async def chat_support(request: Request):
    data = await request.json()
    message = data.get("message", "")
    try:
        api_key = os.getenv("AGENTROUTER_API_KEY") or os.getenv("GROQ_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            return JSONResponse({"reply": "AI support is currently unavailable."})
        prompt = (
            f"You are a helpful agricultural support assistant for 'CropOracle', "
            f"an AI-powered crop recommendation platform for Indian farmers. "
            f"The user asks: '{message}'. "
            f"Keep your reply brief (2-4 sentences), polite, and strictly related to "
            f"farming, crops, soil, or this application."
        )
        text = gemini_generate(prompt, api_key, timeout=5)
        if not text:
            return JSONResponse({"reply": "I'm a bit busy right now. Please try again in a moment!"})
        return JSONResponse({"reply": text})
    except Exception:
        return JSONResponse({"reply": "Sorry, I'm experiencing a temporary issue. Please try again."})

_SUGGEST_CACHE = {}

@app.get("/api/suggest-crops")
def suggest_crops(region: str, state: str, city: str):
    cache_key = f"{city}_{state}_{region}".lower()
    if cache_key in _SUGGEST_CACHE:
        return JSONResponse(_SUGGEST_CACHE[cache_key])

    try:
        api_key = os.getenv("AGENTROUTER_API_KEY") or os.getenv("GROQ_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            return JSONResponse([])

        from .prediction_service import load_all_models
        _, _, _, _, label_encoder, _, _ = load_all_models()
        available_crops = list(label_encoder.classes_)
        crop_list_str = ", ".join(available_crops)

        prompt = f"""
        Act as an expert Indian agricultural database. For the city '{city}', state '{state}' ({region} region),
        suggest 3 suitable crops. You MUST ONLY choose from this list: [{crop_list_str}].
        Do NOT suggest any crop outside this list. Return ONLY a raw JSON array of objects without markdown.
        Format: [{{"crop": "Name", "N": integer, "P": integer, "K": integer, "temperature": float, "humidity": float, "ph": float, "rainfall": float}}]
        """
        raw_text = gemini_generate(prompt, api_key, timeout=15)
        if not raw_text:
            return JSONResponse([])
        clean_text = raw_text.replace('```json', '').replace('```', '').strip()
        import re
        json_str = re.search(r'\[.*\]', clean_text, re.DOTALL).group()
        data = json.loads(json_str)

        available_lower = {c.lower(): c for c in available_crops}
        filtered = []
        for item in data:
            crop_name = item.get("crop", "").strip()
            if crop_name.lower() in available_lower:
                item["crop"] = available_lower[crop_name.lower()]
                if item["crop"] in _CROP_CSV_MEANS:
                    csv_vals = _CROP_CSV_MEANS[item["crop"]]
                    for key in ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]:
                        if key in csv_vals:
                            item[key] = csv_vals[key]
                filtered.append(item)

        _SUGGEST_CACHE[cache_key] = filtered
        return JSONResponse(filtered)
    except Exception as e:
        print("API Suggest Error:", e)
        return JSONResponse([])

# Precompute CSV mean values per crop for autofill
_CROP_CSV_MEANS = {}

def _load_crop_csv_means():
    global _CROP_CSV_MEANS
    try:
        csv_path = Path(__file__).resolve().parent / "data" / "crop_recommendation.csv"
        if csv_path.exists():
            import pandas as pd
            df = pd.read_csv(csv_path)
            features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
            means = df.groupby("label")[features].mean().round(2)
            _CROP_CSV_MEANS = means.to_dict(orient="index")
            print(f"[CSV] Loaded mean values for {len(_CROP_CSV_MEANS)} crops")
    except Exception as e:
        print(f"[CSV] Failed to load crop means: {e}")

_load_crop_csv_means()

@app.get("/api/crop-defaults")
def crop_defaults(crop: str):
    crop_lower = crop.lower().strip()
    for key, vals in _CROP_CSV_MEANS.items():
        if key.lower() == crop_lower:
            return JSONResponse({"crop": key, **vals})
    return JSONResponse({})

@app.get("/api/check-username")
def check_username_api(username: str, db: Session = Depends(get_db)):
    if not username:
        return JSONResponse({"available": False, "error": "Username is required."})
    is_valid, error_msg = validate_username(username)
    if not is_valid:
        return JSONResponse({"available": False, "error": error_msg})
    user = db.query(User).filter(User.username == username).first()
    if user:
        return JSONResponse({"available": False, "error": "Username is already taken."})
    return JSONResponse({"available": True, "error": ""})

# ----------------------------
# DASHBOARD & PREDICTION
# ----------------------------
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request, db: Session = Depends(get_db)):
    user_id = request.cookies.get("user_id")
    if not user_id:
        return RedirectResponse(url="/login", status_code=303)
    user = db.query(User).filter(User.id == int(user_id)).first()
    if not user:
        response = RedirectResponse(url="/login", status_code=303)
        response.delete_cookie("user_id")
        return response
    hour = datetime.now().hour
    if hour < 12:
        greeting_prefix = "Good Morning,"
    elif hour < 17:
        greeting_prefix = "Good Afternoon,"
    else:
        greeting_prefix = "Good Evening,"
    return templates.TemplateResponse("index.html", {
        "request": request,
        "name": f"{user.first_name} {user.last_name}",
        "username": user.username,
        "greeting_prefix": greeting_prefix,
        "is_logged_in": True
    })

@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    N: float = Form(...), P: float = Form(...), K: float = Form(...),
    temperature: float = Form(...), humidity: float = Form(...),
    ph: float = Form(...), rainfall: float = Form(...),
    region: str = Form(...), state: str = Form(...), city: str = Form("Unknown"),
    user_id: str = Cookie(None), db: Session = Depends(get_db)
):
    if not user_id:
        return RedirectResponse(url="/login", status_code=303)
    input_data = {
        "N": N, "P": P, "K": K,
        "temperature": temperature, "humidity": humidity,
        "ph": ph, "rainfall": rainfall,
        "region": region, "state": state, "city": city
    }

    result = predict_crop(input_data)

    try:
        new_log = PredictionLog(
            N=N, P=P, K=K, temperature=temperature, humidity=humidity,
            ph=ph, rainfall=rainfall, region=region, state=state,
            market_price=result.get("market_price", 0),
            production_cost=result.get("production_cost", 0),
            predicted_crop=result["top1"], confidence=result["confidence"],
            risk_score=result["risk_score"], expected_profit=result["expected_profit"],
            user_id=int(user_id)
        )
        db.add(new_log)
        db.commit()
    except Exception as e:
        print(f"[Predict Log Error] {e}")
        db.rollback()

    return templates.TemplateResponse("result.html", {"request": request, "result": result, "is_logged_in": True})

# ----------------------------
# LOGS, ANALYTICS & DOWNLOADS
# ----------------------------
@app.get("/logs", response_class=HTMLResponse)
def view_logs(request: Request, user_id: str = Cookie(None), db: Session = Depends(get_db)):
    if not user_id:
        return RedirectResponse(url="/login", status_code=303)
    user = db.query(User).filter(User.id == int(user_id)).first()
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    logs = db.query(PredictionLog).filter(
        PredictionLog.user_id == int(user_id)
    ).order_by(PredictionLog.id.desc()).all()
    crop_counts = {}
    for log in logs:
        crop_counts[log.predicted_crop] = crop_counts.get(log.predicted_crop, 0) + 1
    return templates.TemplateResponse("logs.html", {
        "request": request, "logs": logs,
        "username": user.username,
        "name": f"{user.first_name} {user.last_name}",
        "is_logged_in": True,
        "chart_labels": list(crop_counts.keys()),
        "chart_data": list(crop_counts.values())
    })

@app.get("/analytics", response_class=HTMLResponse)
def analytics_page(request: Request, user_id: str = Cookie(None), db: Session = Depends(get_db)):
    if not user_id:
        return RedirectResponse(url="/login", status_code=303)
    user = db.query(User).filter(User.id == int(user_id)).first()
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse("analytics.html", {
        "request": request,
        "username": user.username,
        "name": f"{user.first_name} {user.last_name}",
        "is_logged_in": True
    })

@app.get("/view-log/{log_id}", response_class=HTMLResponse)
def view_single_log(request: Request, log_id: int, db: Session = Depends(get_db)):
    user_id = request.cookies.get("user_id")
    if not user_id:
        return RedirectResponse(url="/login", status_code=303)
    log = db.query(PredictionLog).filter(
        PredictionLog.id == log_id,
        PredictionLog.user_id == int(user_id)
    ).first()
    if not log:
        return RedirectResponse(url="/logs", status_code=303)
    input_data = {
        "N": log.N, "P": log.P, "K": log.K,
        "temperature": log.temperature, "humidity": log.humidity,
        "ph": log.ph, "rainfall": log.rainfall,
        "region": log.region or "North", "state": log.state or ""
    }
    result = predict_crop(input_data)
    return templates.TemplateResponse("result.html", {"request": request, "result": result, "is_logged_in": True})

@app.post("/delete-log/{log_id}")
def delete_log(log_id: int, request: Request, db: Session = Depends(get_db)):
    user_id = request.cookies.get("user_id")
    if user_id:
        log = db.query(PredictionLog).filter(
            PredictionLog.id == log_id,
            PredictionLog.user_id == int(user_id)
        ).first()
        if log:
            db.delete(log)
            db.commit()
    return RedirectResponse(url="/logs", status_code=303)

@app.post("/clear-logs")
def clear_logs(request: Request, db: Session = Depends(get_db)):
    user_id = request.cookies.get("user_id")
    if user_id:
        db.query(PredictionLog).filter(PredictionLog.user_id == int(user_id)).delete()
        db.commit()
    return RedirectResponse(url="/logs", status_code=303)

@app.get("/download_csv")
def download_csv(user_id: str = Cookie(None), db: Session = Depends(get_db)):
    if not user_id:
        return RedirectResponse(url="/login", status_code=303)
    logs = db.query(PredictionLog).filter(PredictionLog.user_id == int(user_id)).all()
    file_path = "prediction_logs.csv"
    with open(file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["N", "P", "K", "Temperature", "Humidity", "pH", "Rainfall", "Region", "State", "Crop", "Confidence"])
        for log in logs:
            writer.writerow([log.N, log.P, log.K, log.temperature, log.humidity, log.ph, log.rainfall, log.region, log.state, log.predicted_crop, log.confidence])
    return FileResponse(file_path, media_type='text/csv', filename="croporacle_prediction_logs.csv")

@app.get("/download_pdf")
def download_pdf(user_id: str = Cookie(None), db: Session = Depends(get_db)):
    if not user_id:
        return RedirectResponse(url="/login", status_code=303)
    logs = db.query(PredictionLog).filter(PredictionLog.user_id == int(user_id)).all()
    file_path = "prediction_logs.pdf"
    doc = SimpleDocTemplate(file_path)
    elements = []
    styles = getSampleStyleSheet()
    for log in logs:
        text = f"Crop: {log.predicted_crop} — Confidence: {log.confidence}% | Region: {log.region}, {log.state}"
        elements.append(Paragraph(text, styles["Normal"]))
        elements.append(Spacer(1, 12))
    doc.build(elements)
    return FileResponse(file_path, media_type='application/pdf', filename="croporacle_prediction_logs.pdf")