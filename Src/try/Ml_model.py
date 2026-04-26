import numpy as np
import pandas as pd
import optuna

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# ---------------- LOAD DATA ----------------
df = pd.read_csv("Data/cleaned.csv")

X = df.drop("Exited", axis=1)
y = df["Exited"]


# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ---------------- SCALING ----------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ---------------- OPTUNA OBJECTIVE ----------------
def objective(trial):

    # Split training into train + validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=42
    )

    model_name = trial.suggest_categorical(
        "model", ["logreg", "svm", "rf", "xgb"]
    )

    threshold = trial.suggest_float("threshold", 0.2, 0.7)

    # -------- Logistic Regression --------
    if model_name == "logreg":
        model = LogisticRegression(
            C=trial.suggest_float("C", 1e-3, 10, log=True),
            solver=trial.suggest_categorical("solver", ["liblinear", "lbfgs"]),
            class_weight="balanced",
            max_iter=1000
        )

    # -------- SVM --------
    elif model_name == "svm":
        model = SVC(
            C=trial.suggest_float("C", 1e-3, 10, log=True),
            kernel=trial.suggest_categorical("kernel", ["linear", "rbf"]),
            gamma=trial.suggest_categorical("gamma", ["scale", "auto"]),
            probability=True,
            class_weight="balanced"
        )

    # -------- Random Forest --------
    elif model_name == "rf":
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            max_depth=trial.suggest_int("max_depth", 3, 20),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
            class_weight="balanced",
            random_state=42
        )

    # -------- XGBoost --------
    elif model_name == "xgb":
        scale_pos_weight = (len(y_tr) - sum(y_tr)) / sum(y_tr)

        model = XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=42
        )

    # Train model
    model.fit(X_tr, y_tr)

    # Predict probabilities
    y_probs = model.predict_proba(X_val)[:, 1]

    # Apply threshold
    y_pred = (y_probs > threshold).astype(int)

    # Optimize F1 score
    return f1_score(y_val, y_pred)


# ---------------- RUN OPTUNA ----------------
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print("\nBest Parameters:")
print(study.best_params)

print("\nBest F1 Score:")
print(study.best_value)


# ---------------- FINAL MODEL ----------------
best = study.best_params

if best["model"] == "logreg":
    final_model = LogisticRegression(
        C=best["C"],
        solver=best["solver"],
        class_weight="balanced",
        max_iter=1000
    )

elif best["model"] == "svm":
    final_model = SVC(
        C=best["C"],
        kernel=best["kernel"],
        gamma=best["gamma"],
        probability=True,
        class_weight="balanced"
    )

elif best["model"] == "rf":
    final_model = RandomForestClassifier(
        n_estimators=best["n_estimators"],
        max_depth=best["max_depth"],
        min_samples_split=best["min_samples_split"],
        min_samples_leaf=best["min_samples_leaf"],
        class_weight="balanced",
        random_state=42
    )

elif best["model"] == "xgb":
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

    final_model = XGBClassifier(
        n_estimators=best["n_estimators"],
        max_depth=best["max_depth"],
        learning_rate=best["learning_rate"],
        subsample=best["subsample"],
        colsample_bytree=best["colsample_bytree"],
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )


# Train final model
final_model.fit(X_train, y_train)

# Final prediction
y_probs = final_model.predict_proba(X_test)[:, 1]
y_pred = (y_probs > best["threshold"]).astype(int)


# ---------------- EVALUATION ----------------
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# this ouput
# Accuracy: 83% (good)
# F1-score: 0.63 (strong for imbalance)

# using  
# Best Parameters:
# {'model': 'rf', 'threshold': 0.4249525320160749, 'n_estimators': 190, 'max_depth': 16, 'min_samples_split': 7, 'min_samples_leaf': 4}
