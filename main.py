from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from PIL import Image
import io
import hashlib
import time
import os
import uuid

from fingerprint import compute_phash
from frequency import compute_fft_score
from noise import compute_noise_score
from model import predict_ai_probability, generate_gradcam, extract_embedding
from anomaly import get_anomaly_score


app = FastAPI(title="TARA Core Engine v4.0 - Production Stable")

# ===============================
# Mount Static + Templates
# ===============================
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/explanations", StaticFiles(directory="explanations"), name="explanations")
templates = Jinja2Templates(directory="templates")


# ===============================
# Utility
# ===============================
def compute_sha256(image_bytes):
    sha = hashlib.sha256()
    sha.update(image_bytes)
    return sha.hexdigest()


# ===============================
# Confidence Calibration
# ===============================
def calibrate_confidence(prob):
    temperature = 1.3
    calibrated = prob ** (1 / temperature)
    return round(min(calibrated * 100, 99.5), 2)


# ===============================
# Core Evaluation Engine
# ===============================
def evaluate_image(image):

    # ===============================
    # 1. CNN Prediction
    # ===============================
    probs = predict_ai_probability(image)
    fake_probability = float(probs[0])
    real_probability = float(probs[1])

    # ===============================
    # 2. Embedding + Anomaly
    # ===============================
    embedding = extract_embedding(image)
    anomaly_score = float(get_anomaly_score(embedding))

    # ===============================
    # 3. Forensic Signals
    # ===============================
    fft_score = float(compute_fft_score(image))
    noise_score = float(compute_noise_score(image))

    # ===============================
    # 4. Bias-Corrected Fusion Logic
    # ===============================

    # ---- Strong FAKE trigger (lowered threshold)
    if fake_probability > 0.45:
        predicted_label = "FAKE"
        combined_score = fake_probability

    # ---- Strong REAL trigger (harder to trust)
    elif real_probability > 0.90 and anomaly_score < 0.30:
        predicted_label = "REAL"
        combined_score = fake_probability

    # ---- Fusion Zone (most cases)
    else:
        combined_score = (
            0.50 * fake_probability +
            0.25 * anomaly_score +
            0.15 * fft_score +
            0.10 * noise_score
        )

        if combined_score > 0.48:
            predicted_label = "FAKE"
        elif combined_score < 0.30:
            predicted_label = "REAL"
        else:
            predicted_label = "UNCERTAIN"

    # ===============================
    # 5. Risk Level
    # ===============================
    if combined_score > 0.70:
        risk_level = "High Risk"
    elif combined_score > 0.45:
        risk_level = "Suspicious"
    else:
        risk_level = "Safe"

    # ===============================
    # 6. Confidence (Calibrated)
    # ===============================
    raw_conf = max(fake_probability, real_probability)
    confidence = calibrate_confidence(raw_conf)

    return {
        "predicted_label": predicted_label,
        "fake_probability": round(fake_probability, 4),
        "real_probability": round(real_probability, 4),
        "confidence": confidence,
        "risk_score": round(combined_score, 4),
        "risk_level": risk_level,
        "fft_score": round(fft_score, 4),
        "noise_score": round(noise_score, 4),
        "anomaly_score": round(anomaly_score, 4)
    }

# ===============================
# Context Interpretation Layer
# ===============================
def interpret_case(evaluation, case_type):

    label = evaluation["predicted_label"]
    risk = evaluation["risk_level"]

    case_responses = {
        "food_refund": {
            "FAKE": "Image appears AI-generated. Refund claim may be fraudulent.",
            "REAL": "Image appears authentic. Refund claim likely valid.",
            "UNCERTAIN": "Image authenticity unclear. Manual verification recommended."
        },
        "document_verification": {
            "FAKE": "Possible document manipulation detected.",
            "REAL": "Document appears authentic.",
            "UNCERTAIN": "Document requires manual inspection."
        },
        "social_media_misuse": {
            "FAKE": "Possible AI impersonation detected.",
            "REAL": "No major manipulation detected.",
            "UNCERTAIN": "Potential manipulation. Further investigation advised."
        },
        "personal_safety": {
            "FAKE": "High manipulation risk detected. Proceed cautiously.",
            "REAL": "No strong manipulation indicators detected.",
            "UNCERTAIN": "Content may require deeper forensic analysis."
        }
    }

    return case_responses.get(case_type, {}).get(label, "Forensic analysis complete.")


# ===============================
# Login Page
# ===============================
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/login", response_class=HTMLResponse)
async def login(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


# ===============================
# Dashboard Page
# ===============================
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


# ===============================
# Home Route
# ===============================
@app.get("/", response_class=HTMLResponse)
async def root_login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


# ===============================
# AI Detection Route
# ===============================
@app.get("/ai", response_class=HTMLResponse)
async def ai_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ===============================
# Morphing Route
# ===============================
@app.get("/morph")
async def morph_page():
    return RedirectResponse(url="http://127.0.0.1:8010/forensics", status_code=302)


# ===============================
# JSON API Endpoint
# ===============================
@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    try:
        start_time = time.time()

        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        sha256_hash = compute_sha256(image_bytes)
        phash_value = compute_phash(image)

        evaluation = evaluate_image(image)
        explanation_path = generate_gradcam(image)

        processing_time = round(time.time() - start_time, 3)

        return JSONResponse({
            "image_fingerprint": {
                "sha256": sha256_hash,
                "phash": str(phash_value)
            },
            "signals": {
                "fft_score": evaluation["fft_score"],
                "noise_score": evaluation["noise_score"],
                "anomaly_score": evaluation["anomaly_score"]
            },
            "ai_model": {
                "predicted_label": evaluation["predicted_label"],
                "confidence_percent": evaluation["confidence"]
            },
            "risk_analysis": {
                "risk_score": evaluation["risk_score"],
                "risk_level": evaluation["risk_level"]
            },
            "processing_time_seconds": processing_time,
            "explanation_image": explanation_path,
            "status": "processed"
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ===============================
# UI Endpoint
# ===============================
@app.post("/analyze-ui", response_class=HTMLResponse)
async def analyze_ui(
    request: Request,
    file: UploadFile = File(...),
    case_type: str = Form(...)
):
    try:
        start_time = time.time()

        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        os.makedirs("static/uploads", exist_ok=True)
        upload_path = f"static/uploads/{uuid.uuid4().hex}.png"
        image.save(upload_path)

        evaluation = evaluate_image(image)
        case_message = interpret_case(evaluation, case_type)
        explanation_path = generate_gradcam(image)

        processing_time = round(time.time() - start_time, 3)

        result = {
            "predicted_label": evaluation["predicted_label"],
            "confidence": evaluation["confidence"],
            "risk_level": evaluation["risk_level"],
            "case_message": case_message,
            "processing_time": processing_time,
            "uploaded_image": "/" + upload_path,
            "explanation_image": "/" + explanation_path
        }

        return templates.TemplateResponse(
            "index.html",
            {"request": request, "result": result}
        )

    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": str(e)}
        )


# ===============================
# ROUTE SELECTION
# ===============================
@app.post("/select-module")
async def select_module(module: str = Form(...)):

    if module == "detection":
        return RedirectResponse(url="/", status_code=302)

    elif module == "morphing":
        # If morphing is on another laptop/server:
        return RedirectResponse(url="http://127.0.0.1:8010", status_code=302)

    return RedirectResponse(url="/dashboard", status_code=302)
