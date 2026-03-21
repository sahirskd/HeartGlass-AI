import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt
import traceback
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

st.set_page_config(page_title="HeartGlass AI", layout="wide", page_icon="🫀")
st.title("🫀 HeartGlass AI: Explainable Heart Disease Prediction")
st.markdown("Enter patient details below to get a risk assessment and an explanation of the driving factors.")

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(base_dir, 'ml/models/heart_disease_model.joblib')

@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

def get_shap_explainer(model_obj, X_transformed):
    """Returns the appropriate SHAP explainer based on model type."""
    if isinstance(model_obj, (RandomForestClassifier, GradientBoostingClassifier)):
        return shap.TreeExplainer(model_obj)
    elif isinstance(model_obj, LogisticRegression):
        background = np.zeros((1, X_transformed.shape[1]))
        return shap.LinearExplainer(model_obj, background)
    else:
        try:
            return shap.Explainer(model_obj, X_transformed)
        except Exception:
            return shap.Explainer(model_obj.predict, X_transformed)

try:
    artifacts = load_artifacts()
    model_pipeline = artifacts['model']
    feature_names = artifacts['features']
    cat_cols = artifacts['categorical_cols']
    num_cols = artifacts['numerical_cols']
    
    st.sidebar.header("Patient Data")
    
    def user_input_features():
        age = st.sidebar.number_input("Age", 20.0, 100.0, 50.0)
        sex = st.sidebar.selectbox("Sex", ['Male', 'Female'])
        dataset = st.sidebar.selectbox("Dataset Source", ['Cleveland', 'Hungary', 'Switzerland', 'VA Long Beach'])
        cp = st.sidebar.selectbox("Chest Pain Type (cp)", ['typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic'])
        trestbps = st.sidebar.number_input("Resting Blood Pressure (trestbps)", 80.0, 200.0, 120.0)
        chol = st.sidebar.number_input("Cholesterol (chol)", 0.0, 600.0, 200.0)
        fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 (fbs)", ['False', 'True'])
        restecg = st.sidebar.selectbox("Resting ECG (restecg)", ['normal', 'st-t abnormality', 'lv hypertrophy'])
        thalch = st.sidebar.number_input("Max Heart Rate (thalch)", 60.0, 230.0, 150.0)
        exang = st.sidebar.selectbox("Exercise Induced Angina (exang)", ['False', 'True'])
        oldpeak = st.sidebar.number_input("ST Depression (oldpeak)", -3.0, 7.0, 1.0, 0.1)
        slope = st.sidebar.selectbox("Slope of Peak ST Segment (slope)", ['upsloping', 'flat', 'downsloping'])
        ca = st.sidebar.number_input("Major Vessels (ca)", 0.0, 4.0, 0.0, 1.0)
        thal = st.sidebar.selectbox("Thalassemia (thal)", ['normal', 'fixed defect', 'reversable defect'])

        data = {
            'age': age, 'sex': sex, 'dataset': dataset, 'cp': cp,
            'trestbps': trestbps, 'chol': chol, 'fbs': fbs, 'restecg': restecg,
            'thalch': thalch, 'exang': exang, 'oldpeak': oldpeak, 'slope': slope,
            'ca': ca, 'thal': thal
        }
        return pd.DataFrame([data])

    input_df = user_input_features()
    st.subheader("Patient Input Summary")
    st.dataframe(input_df, hide_index=True)

    if st.button("Predict Chronic Disease Risk", type="primary"):
        with st.spinner("Analyzing patient data..."):
            prob = model_pipeline.predict_proba(input_df)[0][1]
            pred_class = int(model_pipeline.predict(input_df)[0])
            
            st.divider()
            st.subheader("Risk Assessment")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                risk_status = "⚠️ High Risk" if prob > 0.5 else "✅ Low Risk"
                st.markdown(f"### {risk_status}")
                st.metric(label="Risk Probability", value=f"{prob:.1%}")
                
                if prob > 0.7:
                    st.warning("Urgent clinical follow-up is recommended.")
                elif prob > 0.5:
                    st.info("Further diagnostic tests are advised.")
                else:
                    st.success("Patient profile appears stable.")

            with col2:
                st.subheader("Key Risk Drivers")
                preprocessor = model_pipeline.named_steps['preprocessor']
                classifier = model_pipeline.named_steps['model']
                X_transformed = preprocessor.transform(input_df)
                
                try:
                    transformed_feature_names = preprocessor.get_feature_names_out()
                    transformed_feature_names = [name.split('__')[-1] for name in transformed_feature_names]
                except Exception:
                    transformed_feature_names = num_cols + cat_cols
                
                explainer = get_shap_explainer(classifier, X_transformed)
                shap_values = explainer.shap_values(X_transformed)
                
                # Extract SHAP values for the positive class
                if isinstance(shap_values, list):
                    sv = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
                elif hasattr(shap_values, "values"):
                    sv = shap_values.values[0]
                    if len(sv.shape) > 1:
                        sv = sv[:, 1]
                elif len(shap_values.shape) == 3:
                    sv = shap_values[0, :, 1]
                elif len(shap_values.shape) == 2:
                    sv = shap_values[0]
                else:
                    sv = shap_values
                
                actual_names = transformed_feature_names[:len(sv)]
                expl_df = pd.DataFrame({"Feature": actual_names, "Impact": sv})
                expl_df = expl_df.sort_values(by="Impact", key=abs, ascending=False).head(8)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                colors = ['#ff4b4b' if x > 0 else '#00cc96' for x in expl_df["Impact"]]
                ax.barh(expl_df["Feature"], expl_df["Impact"], color=colors)
                ax.set_xlabel("SHAP Value (Risk Contribution)")
                ax.set_title("Contribution to Heart Disease Risk")
                ax.invert_yaxis()
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
                
                st.pyplot(fig)
                st.caption("🔴 Increases Risk | 🟢 Decreases Risk")
                st.markdown("""
                *The SHAP plot shows how each patient feature contributed to the final probability. 
                Values further from zero have a higher impact on the model's decision.*
                """)

except Exception as e:
    st.error(f"Error loading models or generating prediction: {str(e)}")
    st.code(traceback.format_exc(), language="python")
    st.info("🚨 Make sure you have trained and selected a model in `ml/train.py` first!")


