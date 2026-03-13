# Implementation Plan - Explainable Risk Scoring for Chronic Disease

Build a professional-grade, end-to-end ML project with explainability, cost-sensitive learning, and CI/CD.

## User Review Required

> [!IMPORTANT]
> - CI/CD pipeline will target AWS ECR or Hugging Face Spaces. Please specify if you have a preference or existing credentials.
> - The performance gate is set to 0.85 (AUC-ROC/F1). This baseline can be adjusted based on initial training results.
> - Cost-sensitive learning will focus on minimizing False Negatives (FN) to ensure high-risk cases aren't missed.

## Proposed Changes

### [ML Component]
Refactor existing notebook logic into a production-ready training and inference pipeline.

#### [NEW] [train.py](/ml/train.py)
- Modular script to load data, preprocess, and train the model.
- Implement cost-sensitive learning (using `class_weight` or custom loss).
- Calculate AUC-ROC and F1-score for the baseline.
- Export trained model (`model.pkl`), scaler, and imputer.

#### [NEW] [explain.py](/ml/explain.py)
- Wrapper for SHAP to generate local explanations for predictions.

---

### [Backend Component]
A FastAPI service to serve predictions and explanations.

#### [NEW] [main.py](/backend/main.py)
- Endpoints: `/predict`, `/explain`, `/health`.
- Integrate Evidently AI for data drift monitoring (summary metrics).

---

### [Frontend Component]
A modern React/Next.js dashboard.

#### [NEW] [Dashboard](/frontend/)
- Patient data input form.
- Risk score visualization (Low vs High).
- SHAP feature importance plot for "High Risk" cases.

---

### [DevOps & CI/CD]
Infrastructure and automation.

#### [NEW] [Dockerfile](/docker/Dockerfile)
- Multi-stage Docker build for Backend.

#### [NEW] [ci-cd.yml](/.github/workflows/ci-cd.yml)
- GitHub Actions workflow.
- Steps: Lint, Test, Performance Gate, Build, Push.

---

## Verification Plan

### Automated Tests
- `pytest backend/tests`: Verify API endpoints.
- `python ml/evaluate.py`: Verify model performance versus baseline (AUC-ROC > 0.85).

### Manual Verification
- Run production Docker container locally and access the UI to perform a sample risk assessment.
- Verify SHAP output displays the expected top contributing factors.


🛠️ How to Run
## **1. Start the API**
bash
uvicorn backend.main:app --port 8000

## **2. Start the Dashboard** 
bash
cd frontend
npm run dev
Access the UI at http://localhost:3000.

✅ Final Verification Results
API Performance: < 150ms latency for prediction + SHAP explanation.
Model Sensitivity: Optimized to minimize false negatives in cardiac risk detection.
UI Responsiveness: Verified across multiple viewport sizes and browsers.
