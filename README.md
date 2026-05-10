# Social Exergames with ML-Based Team Engagement Analytics (Same UI)

This version keeps the original SYNC SQUAD user interface and adds an XGBoost-based backend.

## Why this fixes your previous error
This backend uses the **native XGBoost Booster API** (`xgboost.train`) instead of `XGBRegressor`, so it does **not depend on scikit-learn**.

## Run
```bash
cd social_exergames_same_ui_xgboost
pip install -r requirements.txt
uvicorn app:app --reload
```

Open:
```bash
http://127.0.0.1:8000
```

## API
- `GET /api/health`
- `POST /api/predict`
- `POST /api/train-demo`

## Note
The frontend is the same original UI. The backend is now ML-based with XGBoost and ready for integrating live frontend feature extraction.
