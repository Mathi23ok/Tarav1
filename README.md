# TARA v1 — Multi-Signal Image Forensics System

v1: rule-based multi-signal system (EfficientNet + heuristics)  
v2: learned fusion system (ResNet + FusionNet)

## Overview
TARA is a system for detecting AI-generated or manipulated images using deep learning + forensic signals instead of relying on a single model.

---

## Key Idea
TARA combines multiple signals:
- CNN prediction (EfficientNet)
- Anomaly detection (embeddings)
- Frequency analysis (FFT)
- Compression artifacts (ELA)

---

## Pipeline
Image → CNN → Signals (FFT + Noise + Anomaly) → Fusion → Output

---

## Features
- Multi-signal decision system  
- Handles uncertainty (REAL / FAKE / UNCERTAIN)  
- Risk scoring + confidence calibration  
- Grad-CAM explainability  

---

## Performance
- ROC-AUC: ~0.96  
- Tested on real and AI-generated datasets  
- Includes noisy / real-world cases  

---

## API
POST /analyze  

Example response:
{
  "predicted_label": "FAKE",
  "confidence_percent": 87.3,
  "risk_level": "High Risk"
}

---

## Run Locally
pip install -r requirements.txt  
uvicorn main:app --reload  

---

## Tech Stack
- Python, FastAPI  
- PyTorch (EfficientNet)  
- OpenCV, NumPy  

---

## Limitations
- Dataset bias affects generalization  
- Fusion is heuristic  
- Not tested for adversarial attacks  

---

## Summary
TARA demonstrates a system-level approach to AI detection by combining multiple signals for better real-world robustness.
