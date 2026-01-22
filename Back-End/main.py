from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
import shutil

from AutoML.Data_Pre_Processing.Data_quality_engine import DataQualityEngine

app = FastAPI()

# ===================== CORS =====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== STORAGE =====================
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

LAST_FILE_PATH = None

# ===================== ANALYZE =====================
@app.post("/data-quality/analyze")
async def analyze_data(file: UploadFile = File(...)):
    global LAST_FILE_PATH

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    LAST_FILE_PATH = file_path

    df = pd.read_csv(file_path)

    engine = DataQualityEngine(df)
    report = engine.analyze()

    return report

# ===================== APPLY CLEANING =====================
@app.post("/data-quality/apply")
async def apply_cleaning(payload: dict):
    global LAST_FILE_PATH

    if LAST_FILE_PATH is None:
        return {"error": "No dataset analyzed yet"}

    decisions = payload.get("decisions", {})

    df = pd.read_csv(LAST_FILE_PATH)

    engine = DataQualityEngine(df)
    clean_df, report = engine.apply_decisions(decisions)

    return report
