# train_depression.py
import numpy as np
import pandas as pd
import joblib
from pathlib import Patho
import os

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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

LABEL_NAME = "Depression_Label"
FEATURE_PREFIX = "Q"   # we'll auto-pick Q1A..Q42A

# ----------------------------
# Helpers
# ----------------------------
def evaluate_and_print(y_true, y_pred, model_name: str) -> float:
    """Print evaluation metrics and return macro-F1."""
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"\n=== {model_name} ===")
    print("Macro-F1:", macro_f1)
    print("\nClassification Report:\n", classification_report(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    return macro_f1

# ----------------------------
# Main
# ----------------------------
def main():
    # 1) Load cleaned dataset
    df = pd.read_csv(DATA_PATH)
    #df = load_and_prepare_data("C:/Users/rasman khurshid/Downloads/data.csv")

    # 2) Features / target
    feature_cols = [c for c in df.columns if c.startswith(FEATURE_PREFIX) and c.endswith("A")]
    X = df[feature_cols].copy()
    y = df[LABEL_NAME].copy()

    # 3) Sacred split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=TEST_SIZE, random_state=SEED
    )

    # 4) Pipeline: SMOTE -> Scale -> Logistic Regression
    pipe_lr = ImbPipeline([
        ("smote", SMOTE(random_state=SEED, k_neighbors=1)),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, random_state=SEED))
    ])

    # 5) Hyperparameter grid
    param_grid_lr = {
        "clf__C": np.logspace(-2, 2, 5),
        "clf__solver": ["lbfgs", "saga"],
        "clf__penalty": ["l2"],
        "clf__class_weight": [None],   # with SMOTE we generally avoid extra class weighting
    }

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    gs_lr = GridSearchCV(
        estimator=pipe_lr,
        param_grid=param_grid_lr,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        error_score="raise"
    )

    # 6) Fit & evaluate
    gs_lr.fit(X_train, y_train)
    print("Best LR macro-F1 (CV):", gs_lr.best_score_)
    print("Best LR params:", gs_lr.best_params_)

    y_pred = gs_lr.predict(X_test)
    test_macro_f1 = evaluate_and_print(y_test, y_pred, "Depression LR (Test)")

    # 7) Persist the best model
    out_path = BEST_DIR / "depression_best.pkl"
    joblib.dump(gs_lr.best_estimator_, out_path)
    print(f"\n✅ Saved best Depression model to: {out_path.resolve()}")
    print(f"   Test macro-F1: {test_macro_f1:.4f}")

if __name__ == "__main__":
    main()
