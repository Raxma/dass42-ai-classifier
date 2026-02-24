# stress_model_training.py

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# ----------------------------
# Configuration
# ----------------------------
SEED = 42
TEST_SIZE = 0.2
N_SPLITS = 3
DATA_PATH = 'C:/Users/rasman khurshid/Downloads/cleaned_dass_data.csv'
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True, parents=True)

LABEL_NAME = "Stress_Label"
FEATURE_PREFIX = "Q"

# ----------------------------
# Evaluation helper
# ----------------------------
def evaluate_model(y_true, y_pred, model_name):
    """Print classification report and return macro-F1 score."""
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"\n=== {model_name} ===")
    print("Macro-F1:", macro_f1)
    print("\nClassification Report:\n", classification_report(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    return macro_f1

# ----------------------------
# Grid Search Helper
# ----------------------------
def run_grid_search(pipe, param_grid, X_train, y_train, X_test, y_test, name):
    """Train model using GridSearchCV and evaluate on test set."""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        error_score="raise"
    )
    
    gs.fit(X_train, y_train)
    print(f"\nBest params for {name}:", gs.best_params_)
    print("CV macro-F1:", gs.best_score_)

    y_pred = gs.predict(X_test)
    test_macro_f1 = evaluate_model(y_test, y_pred, f"{name} (Test)")
    return gs.best_estimator_, test_macro_f1

# ----------------------------
# Load Data
# ----------------------------
df = pd.read_csv(DATA_PATH)

# Automatically select Q1A–Q42A columns
FEATURE_COLS = [col for col in df.columns if col.startswith(FEATURE_PREFIX) and col.endswith("A")]
X = df[FEATURE_COLS]
y = df[LABEL_NAME]

# Train/Test Split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
)

# ----------------------------
# Define Gradient Boosting Pipeline
# ----------------------------
gb_pipeline = ImbPipeline([
    ("smote", SMOTE(random_state=SEED, k_neighbors=1)),
    ("clf", GradientBoostingClassifier(random_state=SEED))
])

param_grid_gb = {
    "clf__n_estimators": [100, 200],
    "clf__learning_rate": [0.05, 0.1, 0.2],
    "clf__max_depth": [2, 3],
    "clf__subsample": [1.0, 0.8],
}

# ----------------------------
# Train, Evaluate, Save
# ----------------------------
best_model, test_f1 = run_grid_search(
    gb_pipeline,
    param_grid_gb,
    X_train,
    y_train,
    X_test,
    y_test,
    name="GradientBoosting"
)

joblib.dump(best_model, MODEL_DIR / "stress_best.pkl")
print(f"\n🏆 Saved best GradientBoosting model to: {MODEL_DIR / 'stress_best.pkl'}")
