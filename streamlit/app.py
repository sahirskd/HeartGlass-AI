import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt

# Configuration
st.set_page_config(page_title="HeartGlass AI", layout="wide", page_icon="🫀")
st.title("🫀 HeartGlass AI: Explainable Heart Disease Prediction")
st.markdown("Enter patient details below to get a risk assessment and an explanation of the driving factors.")

# Paths setup (allowing the script to run from any directory)
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(base_dir, 'ml/models/heart_disease_model.joblib')
EXPLAINER_PATH = os.path.join(base_dir, 'ml/models/shap_explainer.joblib')

@st.cache_resource
def load_models():
    artifacts = joblib.load(MODEL_PATH)
    explainer = joblib.load(EXPLAINER_PATH)
    return artifacts, explainer

try:
    artifacts, explainer = load_models()
    model = artifacts['model']
    encoder = artifacts['encoder']
    scaler = artifacts['scaler']
    imputer = artifacts['imputer']
    cat_cols = artifacts['cat_cols']
    num_cols = artifacts['num_cols']
    feature_names = artifacts['features']
    
    st.sidebar.header("Patient Data")
    st.sidebar.markdown("Adjust the values to see how they impact the prediction.")
    
    def user_input_features():
        age = st.sidebar.number_input("Age", 20.0, 100.0, 50.0)
        sex = st.sidebar.selectbox("Sex", ['Male', 'Female'])
        dataset = st.sidebar.selectbox("Dataset Source", ['Cleveland', 'Hungary', 'Switzerland', 'VA Long Beach'])
        cp = st.sidebar.selectbox("Chest Pain Type (cp)", ['typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic'])
        trestbps = st.sidebar.number_input("Resting Blood Pressure (trestbps)", 80.0, 200.0, 120.0)
        chol = st.sidebar.number_input("Cholesterol (chol)", 100.0, 600.0, 200.0)
        fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 (fbs)", ['False', 'True'])
        restecg = st.sidebar.selectbox("Resting ECG (restecg)", ['normal', 'st-t abnormality', 'lv hypertrophy'])
        thalch = st.sidebar.number_input("Max Heart Rate (thalch)", 60.0, 220.0, 150.0)
        exang = st.sidebar.selectbox("Exercise Induced Angina (exang)", ['False', 'True'])
        oldpeak = st.sidebar.number_input("ST Depression (oldpeak)", 0.0, 6.0, 1.0, 0.1)
        slope = st.sidebar.selectbox("Slope of Peak ST Segment (slope)", ['upsloping', 'flat', 'downsloping'])
        ca = st.sidebar.number_input("Major Vessels Colored by Fluoroscopy (ca)", 0.0, 4.0, 0.0, 1.0)
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
            # Preprocessing
            df = input_df.copy()
            
            # Use encoder that is fitted with existing categoricals
            # Ensure unseen labels default cleanly if 'unknown_value=-1' was used
            df[cat_cols] = encoder.transform(df[cat_cols])
            df[num_cols] = scaler.transform(df[num_cols])
            
            df_imputed = pd.DataFrame(imputer.transform(df), columns=df.columns)
            
            # Predict
            prob = model.predict_proba(df_imputed)[0][1]
            pred_class = int(model.predict(df_imputed)[0])
            
            # Show Result
            st.divider()
            st.subheader("Risk Assessment")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if prob > 0.5:
                    st.error(f"⚠️ High Risk of Heart Disease")
                    st.metric(label="Risk Probability", value=f"{prob:.1%}")
                else:
                    st.success(f"✅ Low Risk of Heart Disease")
                    st.metric(label="Risk Probability", value=f"{prob:.1%}")
                
            # Explainability
            with col2:
                st.subheader("Why was this decision made?")
                shap_values = explainer.shap_values(df_imputed)
                
                if isinstance(shap_values, list):
                    sv = shap_values[1][0]
                else:
                    if len(shap_values.shape) == 3:
                        sv = shap_values[0, :, 1]
                    else:
                        sv = shap_values[0]
                        
                expl_df = pd.DataFrame({"Feature": feature_names, "Impact": sv})
                expl_df = expl_df.sort_values(by="Impact", key=abs, ascending=False).head(5)
                
                fig, ax = plt.subplots(figsize=(8, 4))
                colors = ['#ff4b4b' if x > 0 else '#00cc96' for x in expl_df["Impact"]]
                ax.barh(expl_df["Feature"], expl_df["Impact"], color=colors)
                ax.set_xlabel("SHAP Value (Impact on prediction)")
                ax.invert_yaxis()  # Top feature at the top
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                st.pyplot(fig)
                st.caption("🔴 Red bars increase the risk score. 🟢 Green bars decrease the risk score.")

except Exception as e:
    import traceback
    st.error(f"Error loading models or generating prediction: {str(e)}")
    st.code(traceback.format_exc(), language="python")
    st.info("🚨 Make sure you have trained models saved in `ml/models/` first! Run `python ml/train.py`.")
