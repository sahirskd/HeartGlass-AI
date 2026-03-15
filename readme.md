# 🫀 HeartGlass AI: Explainable Risk Scoring for Chronic Disease

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://heartglass-ai.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-orange.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-blueviolet.svg)](https://shap.readthedocs.io/en/latest/)

> **"Transforming the Black Box of AI into a Transparent Glass for Healthcare Professionals."**

HeartGlass AI is an end-to-end Machine Learning web application designed to predict the risk of heart disease with **full explainability**. Instead of handing doctors a rigid "black box" prediction, this platform uses **SHAP (SHapley Additive exPlanations)** to break down exactly *why* the model made its decision. This allows medical professionals to trust the model, understand the underlying factors, and make informed clinical decisions.

---

## 🎯 Key Features

- **High-Accuracy Predictions:** Powered by an optimized `XGBoost` classifier trained on comprehensive clinical data.
- **Explainable AI (XAI):** Interactive SHAP force plots that visually explain the impact of each patient feature (e.g., Cholesterol, Max HR, BP) on the final risk score.
- **Interactive UI:** A highly intuitive [Streamlit interface](https://heartglass-ai.streamlit.app) tailored for quick data entry and immediate visual feedback.
- **Production-Ready Backend:** A robust REST API built with `FastAPI`, ready to serve predictions at scale.
- **Data Validation:** Strict input validation utilizing `Pydantic` to ensure clinical data integrity before prediction.

---

## 🏗️ Architecture & Tech Stack

The repository is modularized into distinct components to support future scalability and cloud deployment options (AWS/GCP).

### **Current Stack**
- **Machine Learning**: `scikit-learn`, `xgboost`, `pandas`, `joblib`
- **Explainability**: `shap`, `matplotlib`
- **Backend API**: `FastAPI`, `Uvicorn`, `Pydantic`
- **Frontend App**: `Streamlit`

### **Directory Structure**
```text
├── ml/
│   ├── train.py                 # Model training & pipeline generation script
│   └── models/                  # Serialized joblib artifacts (XGBoost + SHAP explainer)
├── backend/
│   └── main.py                  # FastAPI server exposing the /predict endpoint
├── streamlit/
│   ├── app.py                   # Streamlit frontend application
│   └── requirements.txt         # Pinned frontend dependencies
├── frontend/                    # Under Construction (Next.js/React frontend)
└── data/                        # Dataset & preprocessing components
```

---

## 🚀 Live Demo

Experience the live application here: **[HeartGlass AI on Streamlit](https://heartglass-ai.streamlit.app)**

---

## 💻 Getting Started (Local Development)

Want to run this project locally? Follow these steps to spin up the environment.

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/HeartGlass-AI.git
cd "HeartGlass AI"
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Model Training (Optional)
If you wish to retrain the model and generate new `.joblib` artifacts:
```bash
python3 ml/train.py
```

### 4. Running the Application
You have two ways to run the application locally:

**Option A: Run the Streamlit Interface (Recommended)**
```bash
pip install -r streamlit/requirements.txt
streamlit run streamlit/app.py
```
*Access the UI at `http://localhost:8501`*

**Option B: Run the FastAPI Backend**
```bash
uvicorn backend.main:app --reload
```
*Access the interactive API documentation (Swagger UI) at `http://localhost:8000/docs`*

---

## 🛣️ Future Roadmap

- [x] Train baseline model & integrate SHAP values.
- [x] Build and deploy Streamlit prototype.
- [ ] Migrate the frontend to a high-performance **Next.js** React application.
- [ ] Containerize the full stack using **Docker**.
- [ ] Deploy the microservices architecture on **AWS (EC2/ECS)** or **GCP (Cloud Run)**.
- [ ] Implement MLOps pipelines utilizing **Evidently AI** for data drift monitoring.

---
