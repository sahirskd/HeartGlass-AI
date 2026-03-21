import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn import metrics

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(base_dir, 'ml/data/heart_disease_uci.csv')
MODEL_PATH = os.path.join(base_dir, 'ml/models/heart_disease_model.joblib')
RANDOM_STATE = 32

categorical_cols = ['sex', 'dataset', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
numerical_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', StandardScaler())
        ]), numerical_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_cols)
    ]
)

model_configs = [
    {
        "name": "LogisticRegression",
        "pipeline": Pipeline([
            ('preprocessor', preprocessor),
            ('model', LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight='balanced'))
        ]),
        "params": {
            "model__C": [0.01, 0.1, 1, 10],
            "model__solver": ["liblinear", "lbfgs"]
        }
    },
    {
        "name": "RandomForest",
        "pipeline": Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced'))
        ]),
        "params": {
            "model__n_estimators": [100, 150, 200],
            "model__max_depth": [6, 8, 10],
            "model__min_samples_leaf": [2, 4]
        }
    },
    {
        "name": "GradientBoosting",
        "pipeline": Pipeline([
            ('preprocessor', preprocessor),
            ('model', GradientBoostingClassifier(random_state=RANDOM_STATE))
        ]),
        "params": {
              "model__n_estimators": [100, 150, 200],
              "model__learning_rate": [0.05, 0.08, 0.1],
              "model__max_depth": [2, 3, 5, 7]
        }
    }
]


def train_and_evaluate(X_train, y_train, X_test, y_test):
    results = []
    allModels = []
    best_roc_auc = 0.0
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for config in model_configs:
        print(f"Processing: {config['name']}...")

        grid_search = GridSearchCV(
            estimator=config['pipeline'],
            param_grid=config['params'],
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)

        best_pipe = grid_search.best_estimator_

        prob_train = best_pipe.predict_proba(X_train)[:, 1]
        prob_test = best_pipe.predict_proba(X_test)[:, 1]

        roc_train = metrics.roc_auc_score(y_train, prob_train)
        roc_test = metrics.roc_auc_score(y_test, prob_test)

        results.append({
            "Model": config['name'],
            "Best_Params": grid_search.best_params_,
            "CV_ROC_AUC": grid_search.best_score_,
            "ROC_Train": roc_train * 100,
            "ROC_Test": roc_test * 100,
            "Pipeline_Object": best_pipe
        })

        allModels.append({
            "model_name": config['name'],
            'model_pipe': best_pipe,
        })
        
    results_df = pd.DataFrame(results).sort_values(by="ROC_Test", ascending=False)
    return results_df, allModels

def load_and_preprocess():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    
    df['target'] = (df['num'] > 0).astype(int)
    df = df.drop(columns=['num'])
    
    df['chol'] = df['chol'].replace(0, np.nan)
    df['trestbps'] = df['trestbps'].replace(0, np.nan)
    
    X = df.drop(columns=['target'])
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def train_main():
    X_train, X_test, y_train, y_test = load_and_preprocess()
    
    print("Training and Evaluating Models...")
    summary_report, allModels = train_and_evaluate(X_train, y_train, X_test, y_test)
    
        
    print("\nModel Summary:")
    print(summary_report[['Model', 'CV_ROC_AUC', 'ROC_Train', 'ROC_Test']])
    
    print('\nAvailable Models:')
    for index, model in enumerate(allModels):
        print(index, model['model_name'])
    selected_model_index = input("Select model index: ")
    
    selected_model = allModels[int(selected_model_index)]['model_pipe']

    print("\nBest Model Evaluation:")
    y_pred = selected_model.predict(X_test)
    y_prob = selected_model.predict_proba(X_test)[:, 1]
    
    print(classification_report(y_test, y_pred))
    auc = metrics.roc_auc_score(y_test, y_prob)
    print(f"Final Test AUC-ROC: {auc:.4f}")
    
    if auc < 0.85:
        print("WARNING: Model performance below threshold!")
        
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    artifacts = {
        'model': selected_model,
        'features': X_train.columns.tolist(),
        'categorical_cols': categorical_cols,
        'numerical_cols': numerical_cols
    }
    
    joblib.dump(artifacts, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print("Done.")

if __name__ == "__main__":
    train_main()
