# anxiety_model_training.py

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import os

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# ----------------------------
# Config
# ----------------------------
SEED = 42
TEST_SIZE = 0.2
N_SPLITS = 3
DATA_PATH = Path(os.getenv("DASS_DATA_PATH", "data/cleaned_dass_data.csv"))
BEST_DIR = Path("models")
BEST_DIR.mkdir(exist_ok=True, parents=True)

LABEL_NAME = "Anxiety_Label"
FEATURE_COLS_PREFIX = "Q"   # we’ll select Q1A..Q42A automatically

# ----------------------------
# Helper functions
# ----------------------------
def evaluate_and_print(y_true, y_pred, model_name):
    """Print evaluation metrics and return macro-F1."""
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"\n=== {model_name} ===")
    print("Macro-F1:", macro_f1)
    print("\nClassification Report:\n", classification_report(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    return macro_f1

def run_grid_search(name, pipe, param_grid, X_train, y_train, X_test, y_test):
    """Run GridSearchCV for a given pipeline & param grid, evaluate on test."""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    gs = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        error_score="raise",
    )

    gs.fit(X_train, y_train)
    y_pred = gs.predict(X_test)

    print(f"\nBest params for {name}:", gs.best_params_)
    print("CV macro-F1:", gs.best_score_)

    test_macro_f1 = evaluate_and_print(y_test, y_pred, f"{name} (Test)")
    return gs, test_macro_f1

# ----------------------------
# Load & prepare data
# ----------------------------
df = pd.read_csv(DATA_PATH)

# Auto-pick Q1A..Q42A (or any QxxA present)
FEATURE_COLS = [c for c in df.columns if c.startswith(FEATURE_COLS_PREFIX) and c.endswith("A")]
X = df[FEATURE_COLS].copy()
y = df[LABEL_NAME].copy()

# One sacred split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=TEST_SIZE, random_state=SEED
)

# ----------------------------
# 1) Logistic Regression + SMOTE + Scaling
# ----------------------------
pipe_lr = ImbPipeline([
    ("smote", SMOTE(random_state=SEED, k_neighbors=1)),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=5000, random_state=SEED))
])

param_grid_lr = {
    "clf__C": np.logspace(-2, 2, 5),
    "clf__solver": ["lbfgs", "saga"],
    "clf__penalty": ["l2"],
    "clf__class_weight": [None],   # with SMOTE we usually don't rebalance again
}

lr_model, lr_test_f1 = run_grid_search(
    name="LogisticRegression",
    pipe=pipe_lr,
    param_grid=param_grid_lr,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test
)

joblib.dump(lr_model.best_estimator_, BEST_DIR / "anxiety_logreg.pkl")



# ----------------------------
# Pick and save the winner
# ----------------------------
# Save the best model
model_path = BEST_DIR / "anxiety_best.pkl"
joblib.dump(lr_model.best_estimator_, model_path)
print(f"\n✅ Saved best LogisticRegression model to: {model_path}")
