from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import shap
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Explainable Risk Scoring API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For production, specify the actual origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load artifacts
MODEL_ARTIFACT_PATH = 'ml/models/heart_disease_model.joblib'
EXPLAINER_PATH = 'ml/models/shap_explainer.joblib'

if not os.path.exists(MODEL_ARTIFACT_PATH):
    raise RuntimeError("Model artifacts not found. Please run training first.")

artifacts = joblib.load(MODEL_ARTIFACT_PATH)
model = artifacts['model']
encoder = artifacts['encoder']
scaler = artifacts['scaler']
imputer = artifacts['imputer']
feature_names = artifacts['features']
cat_cols = artifacts['cat_cols']
num_cols = artifacts['num_cols']

explainer = joblib.load(EXPLAINER_PATH)

class PatientData(BaseModel):
    age: float
    sex: str
    dataset: str
    cp: str
    trestbps: float
    chol: float
    fbs: str
    restecg: str
    thalch: float
    exang: str
    oldpeak: float
    slope: str
    ca: float
    thal: str

def preprocess_input(data: PatientData):
    # Convert Pydantic model to DataFrame
    df = pd.DataFrame([data.dict()])
    
    # Preprocessing
    df[cat_cols] = encoder.transform(df[cat_cols])
    df[num_cols] = scaler.transform(df[num_cols])
    
    # Impute missing values (if any)
    df_imputed = pd.DataFrame(imputer.transform(df), columns=df.columns)
    return df_imputed

@app.post("/predict")
async def predict(data: PatientData):
    try:
        X = preprocess_input(data)
        prob = model.predict_proba(X)[0][1]
        prediction = int(model.predict(X)[0])
        
        risk_level = "High" if prob > 0.5 else "Low"
        
        return {
            "probability": float(prob),
            "prediction": prediction,
            "risk_level": risk_level
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain")
async def explain(data: PatientData):
    try:
        X = preprocess_input(data)
        
        # Get SHAP values
        shap_values = explainer.shap_values(X)
        
        # In TreeExplainer for binary RF, shap_values can be a list [class 0, class 1]
        # or just an array depending on SHAP version.
        if isinstance(shap_values, list):
            sv = shap_values[1][0] # Focus on class 1 (High Risk)
        else:
            # For newer SHAP versions / certain models
            if len(shap_values.shape) == 3:
                sv = shap_values[0, :, 1]
            else:
                sv = shap_values[0]

        # Combine feature names with SHAP values
        explanations = []
        for name, value in zip(feature_names, sv):
            explanations.append({"feature": name, "impact": float(value)})
        
        # Sort by absolute impact and take top 5
        explanations.sort(key=lambda x: abs(x['impact']), reverse=True)
        top_explanations = explanations[:5]
        
        return {
            "top_factors": top_explanations,
            "base_value": float(explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
