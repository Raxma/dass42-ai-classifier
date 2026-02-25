# dass42-ai-classifier
# DASS-42 Predictive AI Bot
An automated classification system to identify Depression, Anxiety, and Stress levels using Machine Learning. This repository features a complete Machine Learning ecosystem designed to predict Depression, Anxiety, and Stress levels. By implementing advanced oversampling techniques (SMOTE) and hyperparameter optimization (GridSearchCV), the system overcomes significant dataset imbalances to provide reliable, production-ready diagnostic predictions.
## Quickstart (Run Inference)

### 1) Install
pip install -r requirements.txt

### 2) Ensure models exist
Your `models/` folder must contain:
- `depression_best.pkl`
- `anxiety_best.pkl`
- `stress_best.pkl`

### 3) Run (Python)
Open Python and run:

```python
from predict import predict_all
```
# 42 answers, each must be 1..4
answers = [1]*42

out = predict_all(answers, return_proba=True)
print(out)

## Project Impact (STAR Analysis)
- **Situation:** Built a diagnostic tool based on the DASS-42 psychometric scale.
- **Task:** Address significant class imbalance in mental health data to ensure high-sensitivity predictions for minority classes.
- **Action:** Developed a custom **Imbalanced-learn Pipeline** incorporating **SMOTE** for oversampling, **StandardScaler** for normalization, and **GridSearchCV** for hyperparameter tuning of Gradient Boosting models.
- **Result:** Delivered a production-ready `predict.py` script and validated model files (`.pkl`) achieving high Macro-F1 scores across all psychological states.

## Tech Stack
- **Language:** Python 3.x
- **Core ML:** Scikit-learn, Imbalanced-learn (SMOTE)
- **Deployment:** Joblib for Model Serialization

## Project Structure
- `/models`: Contains serialized `.pkl` files for Stress, Anxiety, and Depression.
- `predict.py`: The core inference script with built-in data validation.
- `stress.py`, `anxiety.py`, `depression.py`: Training scripts and GridSearch logic.
