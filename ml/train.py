import pandas as pd
import numpy as np
import joblib
# import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score
import os

# Configuration
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(base_dir, 'ml/data/heart_disease_uci.csv')
MODEL_PATH = os.path.join(base_dir, 'ml/models/heart_disease_model.joblib')
EXPLAINER_PATH = 'ml/models/shap_explainer.joblib'
RANDOM_STATE = 32

def load_and_preprocess():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    # Drop ID as it's not a feature
    df = df.drop(columns=['id'])
    
    # Define Target: Binary Classification (0: No Disease, 1: Heart Disease)
    # Binary is more suitable for "Risk Scoring" and "False Negative" penalties
    df['target'] = (df['num'] > 0).astype(int)
    df = df.drop(columns=['num'])
    
    # Handle known impossible values as NaN for imputer
    df['chol'] = df['chol'].replace(0, np.nan)
    df['trestbps'] = df['trestbps'].replace(0, np.nan)
    
    # Identify features
    categorical_cols = ['sex', 'dataset', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
    numerical_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
    
    X = df.drop(columns=['target'])
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, categorical_cols, numerical_cols

def train():
    X_train, X_test, y_train, y_test, cat_cols, num_cols = load_and_preprocess()
    
    print("Preproccessing...")
    # Encoding categorical features
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])
    X_test[cat_cols] = encoder.transform(X_test[cat_cols])
    
    # Scaling numerical features
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    
    # Impute missing values
    imputer = KNNImputer(n_neighbors=5)
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
    
    print("Training Model...")
    # Cost-Sensitive Learning: Missing a heart disease (FN) is expensive.
    # Penalize False Negatives by giving higher weight to class 1.
    # In clinical settings, we often weight class 1 significantly higher.
    weights = {0: 1, 1: 3} # 3x penalty for missing a heart disease case
    
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        min_samples_leaf=4,
        class_weight=weights,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # y_pred_train = model.predict(X_train)
    y_prob_train = model.predict_proba(X_train)[:, 1]

    roc_train = roc_auc_score(y_train, y_prob_train)
    roc_test = roc_auc_score(y_test, y_prob)
    
    print("\nModel Evaluation:")
    print(classification_report(y_test, y_pred))
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    print(f"AUC-ROC: {auc:.4f}")
    print(f"F1-Score: {f1:.4f}")

    print('Train Score: ', roc_train)
    print('Test Score: ', roc_test)
    
    # Check baseline (Requirement: > 0.85)
    if auc < 0.85:
        print("WARNING: Model performance below threshold!")
    
    print("\nIntegrating SHAP...")
    # Initialize SHAP explainer
    # TreeExplainer is best for RandomForest
    # explainer = shap.TreeExplainer(model)
    
    # Save artifacts
    print(f"Saving artifacts to ml/models/...")
    artifacts = {
        'model': model,
        'encoder': encoder,
        'scaler': scaler,
        'imputer': imputer,
        'features': X_train.columns.tolist(),
        'cat_cols': cat_cols,
        'num_cols': num_cols
    }
    joblib.dump(artifacts, MODEL_PATH)
    print("Done.")

if __name__ == "__main__":
    train()
