from fastapi import FastAPI
from pydantic import BaseModel, conint
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path


app = FastAPI()
current_dir = Path(__file__).parent


# Allow CORS (for your Flutter app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model & scaler
model_path = current_dir / "models" / "best_model.pkl"  # Updated path
scaler_path = current_dir / "models" / "scaler.pkl"     # Updated path

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Define input data model
class StudentData(BaseModel):
    school: conint(ge=0, le=1)          # 0: GP, 1: MS
    sex: conint(ge=0, le=1)             # 0: F, 1: M
    age: conint(ge=15, le=22)           # Age range
    Medu: conint(ge=0, le=4)            # Mother's education (0-4)
    Fedu: conint(ge=0, le=4)            # Father's education (0-4)
    studytime: conint(ge=1, le=4)       # Weekly study time (1-4)
    failures: conint(ge=0, le=4)        # Past class failures
    famrel: conint(ge=1, le=5)          # Family relationship quality (1-5)
    absences: conint(ge=0, le=93)       # School absences
    G1: conint(ge=0, le=20)             # First period grade (0-20)
    G2: conint(ge=0, le=20)             # Second period grade (0-20)

from sklearn.preprocessing import OneHotEncoder
import pandas as pd

@app.post("/predict")
def predict(data: StudentData):
    try:
        # 1. Create full 32-feature array with correct order
        full_features = np.zeros((1, 32))
        
        # 2. Map your existing 11 features to their positions
        # Using the exact order from debug_features
        feature_mapping = {
            'school': 0,
            'sex': 1,
            'age': 2,
            'Medu': 6,
            'Fedu': 7,
            'studytime': 13,
            'failures': 14,
            'famrel': 23,
            'absences': 29, 
            'G1': 30,
            'G2': 31
        }
        
        # Fill in known features
        for feature, idx in feature_mapping.items():
            full_features[0, idx] = getattr(data, feature)
        
        # 3. Set smart defaults for other features
        # (Adjust these based on your training data statistics)
        defaults = {
            'address': 0,       # 0: urban, 1: rural
            'famsize': 0,       # 0: <=3, 1: >3
            'Pstatus': 0,       # 0: living together, 1: apart
            'Mjob': 2,          # 0: teacher, 1: health, 2: services, 3: at_home, 4: other
            'Fjob': 2,          # same as Mjob
            'reason': 0,        # 0: home, 1: reputation, 2: course, 3: other
            'guardian': 0,      # 0: mother, 1: father, 2: other
            'traveltime': 1,    # 1: <15min, 2: 15-30min, 3: 30min-1h, 4: >1h
            'schoolsup': 0,     # 0: no, 1: yes
            'famsup': 1,        # 0: no, 1: yes
            'paid': 0,          # 0: no, 1: yes
            'activities': 1,    # 0: no, 1: yes
            'nursery': 1,       # 0: no, 1: yes
            'higher': 1,        # 0: no, 1: yes
            'internet': 1,     # 0: no, 1: yes
            'romantic': 0,      # 0: no, 1: yes
            'freetime': 3,      # 1: very low to 5: very high
            'goout': 3,         # same as freetime
            'Dalc': 1,          # 1: very low to 5: very high (workday alcohol)
            'Walc': 1,          # same as Dalc (weekend alcohol)
            'health': 4         # 1: very bad to 5: very good
        }
        
        # Apply defaults
        for feature, value in defaults.items():
            idx = list(scaler.feature_names_in_).index(feature)
            full_features[0, idx] = value
        
        # 4. Scale and predict
        scaled = scaler.transform(full_features)
        prediction = model.predict(scaled)[0]
        
        # 5. Return realistic grade (clamped 0-20)
        final_grade = max(0, min(20, round(float(prediction), 1)))
        return {"predicted_grade": final_grade}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.get("/debug_features")
def debug_features():
    """Reveals what features the model expects"""
    try:
        return {
            "expected_features": list(scaler.feature_names_in_),
            "expected_feature_count": len(scaler.feature_names_in_)
        }
    except Exception as e:
        return {"error": str(e)}