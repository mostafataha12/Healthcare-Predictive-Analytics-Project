from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pickle
import numpy as np
import logging

# ✅ Logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ✅ Load Model
try:
    logger.debug("Loading trained model...")
    model = joblib.load("heart_disease_model.pkl")
    logger.debug("Model loaded.")
except Exception as e:
    logger.error(f"❌ Error loading model: {str(e)}")
    raise

# ✅ Load Feature Names
try:
    logger.debug("Loading model feature names...")
    with open("heart_disease_model.pkl", "rb") as f:
        feature_names = pickle.load(f)
    logger.debug(f"Loaded feature names: {feature_names}")
except Exception as e:
    logger.error(f"❌ Error loading feature names: {str(e)}")
    raise

# ✅ Define FastAPI App
app = FastAPI(title="Heart Disease Predictor API")

# ✅ Define Input Schema
class PatientData(BaseModel):
    Age: float
    Blood_Pressure: float
    Cholesterol_Level: float
    BMI: float
    Triglyceride_Level: float
    CRP_Level: float
    Fasting_Blood_Sugar: float
    # ممكن تزود حقول هنا بنفس أسماء الـ feature_names لو عايز

# ✅ API Home
@app.get("/")
def read_root():
    return {"message": "Welcome to the Heart Disease Predictor API! Visit /docs for API documentation."}

# ✅ Prediction Endpoint
@app.post("/predict/")
def predict_heart_disease(data: PatientData):
    try:
        input_dict = data.dict()
        logger.debug(f"Input received: {input_dict}")

        # تأكد إن كل feature في الموديل له قيمة (وحتى لو ناقص نحط له 0)
        features = [input_dict.get(col, 0) for col in feature_names]
        features_array = np.array([features])
        logger.debug(f"Model input shape: {features_array.shape}")

        # تنبؤ
        prediction = model.predict(features_array)[0]
        result = "Has Heart Disease" if prediction == 1 else "Healthy"
        logger.debug(f"Prediction result: {result}")

        return {"prediction": result}
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return {"error": str(e)}
