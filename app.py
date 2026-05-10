from __future__ import annotations

from pathlib import Path
from typing import Literal

import json
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
MODEL_PATH = BASE_DIR / "engagement_model.json"

FEATURE_COLUMNS = [
    "team_size",
    "avg_reps_per_player",
    "rep_variance",
    "avg_rep_speed",
    "active_time_ratio",
    "posture_score",
    "sync_score",
    "movement_amplitude_mean",
    "movement_amplitude_std",
    "idle_ratio",
    "contribution_balance",
    "session_progress",
]

app = FastAPI(title="Social Exergames with ML-Based Team Engagement Analytics", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

booster: xgb.Booster | None = None
metrics_cache: dict[str, float | int] = {}


class EngagementFeatures(BaseModel):
    team_size: int = Field(ge=1, le=4)
    avg_reps_per_player: float = Field(ge=0, le=250)
    rep_variance: float = Field(ge=0, le=5000)
    avg_rep_speed: float = Field(ge=0, le=5)
    active_time_ratio: float = Field(ge=0, le=1)
    posture_score: float = Field(ge=0, le=100)
    sync_score: float = Field(ge=0, le=100)
    movement_amplitude_mean: float = Field(ge=0, le=2)
    movement_amplitude_std: float = Field(ge=0, le=2)
    idle_ratio: float = Field(ge=0, le=1)
    contribution_balance: float = Field(ge=0, le=1)
    session_progress: float = Field(ge=0, le=1)


class PredictResponse(BaseModel):
    engagement_score: float
    engagement_level: Literal["Low", "Medium", "High"]
    top_factors: list[str]


@app.on_event("startup")
def startup_event() -> None:
    load_or_train_model()


@app.get("/")
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": booster is not None,
        "metrics": metrics_cache,
        "algorithm": "xgboost-native-booster",
    }


@app.post("/api/predict", response_model=PredictResponse)
def predict(payload: EngagementFeatures) -> PredictResponse:
    if booster is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    row = pd.DataFrame([[getattr(payload, col) for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    dmatrix = xgb.DMatrix(row, feature_names=FEATURE_COLUMNS)
    score = float(np.clip(booster.predict(dmatrix)[0], 0, 100))
    return PredictResponse(
        engagement_score=round(score, 2),
        engagement_level=score_to_level(score),
        top_factors=explain_payload(payload),
    )


@app.post("/api/train-demo")
def train_demo() -> dict:
    train_demo_model(save_model=True)
    return {"message": "Demo XGBoost model trained", "metrics": metrics_cache}


@app.get("/api/features")
def feature_spec() -> dict:
    return {"feature_columns": FEATURE_COLUMNS}


def score_to_level(score: float) -> Literal["Low", "Medium", "High"]:
    if score < 40:
        return "Low"
    if score < 70:
        return "Medium"
    return "High"


def explain_payload(payload: EngagementFeatures) -> list[str]:
    reasons: list[str] = []
    if payload.sync_score >= 75:
        reasons.append("Strong team synchronization boosted engagement")
    elif payload.sync_score < 45:
        reasons.append("Weak synchronization reduced the score")

    if payload.posture_score >= 75:
        reasons.append("Good exercise form improved the prediction")
    elif payload.posture_score < 45:
        reasons.append("Poor movement quality lowered the prediction")

    if payload.active_time_ratio >= 0.7 and payload.idle_ratio <= 0.25:
        reasons.append("High active time kept the team engaged")
    elif payload.idle_ratio > 0.35:
        reasons.append("Idle time pulled engagement down")

    if payload.contribution_balance >= 0.7:
        reasons.append("Balanced contribution helped the team score")
    elif payload.contribution_balance < 0.4:
        reasons.append("Uneven participation hurt the score")

    if payload.avg_reps_per_player >= 8:
        reasons.append("Good rep frequency supported engagement")

    if not reasons:
        reasons.append("The score came from the combined pose and team features")
    return reasons[:3]


def load_or_train_model() -> None:
    global booster
    if MODEL_PATH.exists():
        loaded = xgb.Booster()
        loaded.load_model(str(MODEL_PATH))
        booster = loaded
        if not metrics_cache:
            metrics_cache.update({"source": "loaded_saved_model"})
    else:
        train_demo_model(save_model=True)


def train_demo_model(seed: int = 42, save_model: bool = False) -> None:
    global booster, metrics_cache

    rng = np.random.default_rng(seed)
    n = 2600

    df = pd.DataFrame(
        {
            "team_size": rng.integers(1, 5, size=n),
            "avg_reps_per_player": rng.uniform(0, 18, size=n),
            "rep_variance": rng.uniform(0, 36, size=n),
            "avg_rep_speed": rng.uniform(0.05, 1.1, size=n),
            "active_time_ratio": rng.uniform(0.15, 1.0, size=n),
            "posture_score": rng.uniform(25, 100, size=n),
            "sync_score": rng.uniform(10, 100, size=n),
            "movement_amplitude_mean": rng.uniform(0.08, 1.55, size=n),
            "movement_amplitude_std": rng.uniform(0.0, 0.7, size=n),
            "idle_ratio": rng.uniform(0.0, 0.8, size=n),
            "contribution_balance": rng.uniform(0.1, 1.0, size=n),
            "session_progress": rng.uniform(0.05, 1.0, size=n),
        }
    )

    target = (
        12
        + 1.8 * df["avg_reps_per_player"]
        + 14.5 * df["active_time_ratio"]
        + 0.24 * df["posture_score"]
        + 0.22 * df["sync_score"]
        + 10.5 * df["contribution_balance"]
        + 5.5 * df["session_progress"]
        + 4.5 * np.minimum(df["team_size"], 3)
        + 4.0 * df["avg_rep_speed"]
        + 5.0 * df["movement_amplitude_mean"]
        - 0.34 * df["rep_variance"]
        - 10.5 * df["idle_ratio"]
        - 8.0 * df["movement_amplitude_std"]
    )
    target += np.where(
        (df["sync_score"] > 70) & (df["posture_score"] > 70) & (df["active_time_ratio"] > 0.65),
        7.5,
        0.0,
    )
    target += rng.normal(0, 3.0, size=n)
    target = np.clip(target, 0, 100)

    shuffled = df.copy()
    shuffled["target"] = target
    shuffled = shuffled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    split = int(0.8 * len(shuffled))
    train_df = shuffled.iloc[:split]
    test_df = shuffled.iloc[split:]

    dtrain = xgb.DMatrix(train_df[FEATURE_COLUMNS], label=train_df["target"], feature_names=FEATURE_COLUMNS)
    dtest = xgb.DMatrix(test_df[FEATURE_COLUMNS], label=test_df["target"], feature_names=FEATURE_COLUMNS)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.06,
        "max_depth": 5,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "lambda": 1.0,
        "seed": seed,
        "tree_method": "hist",
    }

    evals_result: dict = {}
    trained = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=140,
        evals=[(dtrain, "train"), (dtest, "test")],
        evals_result=evals_result,
        verbose_eval=False,
    )

    preds = np.clip(trained.predict(dtest), 0, 100)
    y_test = test_df["target"].to_numpy()
    mae = float(np.mean(np.abs(preds - y_test)))
    rmse = float(np.sqrt(np.mean((preds - y_test) ** 2)))
    ss_res = float(np.sum((y_test - preds) ** 2))
    ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot else 0.0

    booster = trained
    metrics_cache = {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "r2": round(r2, 3),
        "rows": int(n),
        "source": "synthetic_demo_data",
        "last_test_rmse": round(float(evals_result["test"]["rmse"][-1]), 3),
    }

    if save_model:
        trained.save_model(str(MODEL_PATH))
        (BASE_DIR / "model_meta.json").write_text(json.dumps(metrics_cache, indent=2), encoding="utf-8")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
