"""
FastAPI server exposing the battery cycle life prediction pipeline.

Endpoints
    GET  /api/health           - liveness probe + model/norm availability
    POST /api/predict          - upload a .mat file, get predictions
    POST /api/train            - trigger a training run (synchronous, for demos)
    GET  /api/results/report   - last evaluation report JSON
    GET  /api/results/plots    - list saved plots
    GET  /api/results/plots/{name} - serve a specific plot image
"""

from __future__ import annotations

import os
import sys
import tempfile
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Make the backend package importable when running `uvicorn api.server:app` from backend/
_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from config import Config  # noqa: E402
from main import BatteryCycleLifePipeline  # noqa: E402
from utils import setup_logging  # noqa: E402

logger = setup_logging(level=Config.LOG_LEVEL)

app = FastAPI(
    title="Battery Cycle Life Prediction API",
    version="1.0.0",
    description="REST wrapper around the CNN-based battery RUL model.",
)

# CORS for the Vite dev server (and any other local origins you add).
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Reuse a single pipeline instance — it holds the model + preprocessor state.
_pipeline: Optional[BatteryCycleLifePipeline] = None


def get_pipeline() -> BatteryCycleLifePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = BatteryCycleLifePipeline()
    return _pipeline


class HealthResponse(BaseModel):
    status: str
    model_available: bool
    norm_params_available: bool
    model_path: str
    norm_path: str


class PredictResponse(BaseModel):
    count: int
    predictions: List[float]
    min: float
    max: float
    mean: float


class TrainRequest(BaseModel):
    use_synthetic: bool = True
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    create_plots: bool = True


class TrainResponse(BaseModel):
    success: bool
    metrics: Optional[dict] = None


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    cfg = Config()
    return HealthResponse(
        status="ok",
        model_available=os.path.exists(cfg.MODEL_SAVE_PATH),
        norm_params_available=os.path.exists(cfg.NORM_PARAMS_PATH),
        model_path=cfg.MODEL_SAVE_PATH,
        norm_path=cfg.NORM_PARAMS_PATH,
    )


@app.post("/api/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    if not file.filename or not file.filename.lower().endswith(".mat"):
        raise HTTPException(status_code=400, detail="Upload must be a .mat file")

    pipeline = get_pipeline()
    cfg = pipeline.config

    if not os.path.exists(cfg.MODEL_SAVE_PATH):
        raise HTTPException(status_code=409, detail="Model not trained yet. Run /api/train first.")
    if not os.path.exists(cfg.NORM_PARAMS_PATH):
        raise HTTPException(status_code=409, detail="Normalization params missing. Retrain to regenerate.")

    # Persist upload to a temp file so scipy.io.loadmat can read from disk.
    with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        predictions = pipeline.predict_from_mat(mat_path=tmp_path, save_csv=False)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    values = predictions.tolist()
    return PredictResponse(
        count=len(values),
        predictions=values,
        min=float(min(values)),
        max=float(max(values)),
        mean=float(sum(values) / len(values)),
    )


@app.post("/api/train", response_model=TrainResponse)
def train(req: TrainRequest) -> TrainResponse:
    pipeline = get_pipeline()
    try:
        success = pipeline.run_complete_pipeline(
            use_synthetic=req.use_synthetic,
            download_real=not req.use_synthetic,
            create_plots=req.create_plots,
            save_results=True,
            epochs=req.epochs,
            batch_size=req.batch_size,
        )
    except Exception as e:
        logger.exception("Training failed")
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")

    return TrainResponse(
        success=bool(success),
        metrics=pipeline.results.get("test_metrics"),
    )


@app.get("/api/results/report")
def get_report():
    cfg = Config()
    path = os.path.join(cfg.RESULTS_DIR, "evaluation_report.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No evaluation report found — train first.")
    return FileResponse(path, media_type="application/json")


@app.get("/api/results/plots")
def list_plots():
    cfg = Config()
    if not os.path.isdir(cfg.RESULTS_DIR):
        return JSONResponse(content={"plots": []})
    plots = [f for f in sorted(os.listdir(cfg.RESULTS_DIR)) if f.lower().endswith(".png")]
    return {"plots": plots}


@app.get("/api/results/plots/{name}")
def get_plot(name: str):
    cfg = Config()
    # Basic path-traversal guard.
    safe = os.path.basename(name)
    path = os.path.join(cfg.RESULTS_DIR, safe)
    if not os.path.isfile(path) or not safe.lower().endswith(".png"):
        raise HTTPException(status_code=404, detail=f"Plot not found: {safe}")
    return FileResponse(path, media_type="image/png")
