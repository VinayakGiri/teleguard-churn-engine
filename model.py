import pandas as pd
import numpy as np
import json
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Resolve the project root based on this file's location (src/ -> project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def train_and_evaluate(data_path=None):
    """
    Loads cleaned Telco churn data, trains three ML models, evaluates each one,
    saves the best model and supporting artifacts to the models/ directory.
    """
    # Resolve absolute paths so this script works from any terminal location
    if data_path is None:
        data_path = os.path.join(PROJECT_ROOT, "data", "telco_cleaned.csv")
    models_dir = os.path.join(PROJECT_ROOT, "models")

    # ─────────────────────────────────────────────
    # 1. Load data
    # ─────────────────────────────────────────────
    print(f"Loading data from {data_path}...\n")
    df = pd.read_csv(data_path)

    # ─────────────────────────────────────────────
    # 2. Split features (X) and target (y)
    # ─────────────────────────────────────────────
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    feature_names = list(X.columns)

    # ─────────────────────────────────────────────
    # 3. Train/test split — 80/20, stratified to
    #    preserve the churn ratio in both splits
    # ─────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train size: {len(X_train)} rows | Test size: {len(X_test)} rows\n")

    # ─────────────────────────────────────────────
    # 4. Feature scaling — fit ONLY on training data
    #    to prevent data leakage into the test set
    # ─────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ─────────────────────────────────────────────
    # 5. Define the three candidate models
    # ─────────────────────────────────────────────
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    }

    results = []
    trained_models = {}

    # ─────────────────────────────────────────────
    # 6. Train each model and print evaluation metrics
    # ─────────────────────────────────────────────
    for name, model in models.items():
        print("=" * 55)
        print(f"  Model: {name}")
        print("=" * 55)

        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        print(f"  Accuracy : {acc:.4f}")
        print(f"  AUC-ROC  : {auc:.4f}")
        print(f"\n  Classification Report:\n")
        print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

        # Store results for comparison and JSON export
        results.append({"model": name, "accuracy": round(acc, 4), "auc": round(auc, 4)})
        trained_models[name] = model

    # ─────────────────────────────────────────────
    # 7. Pick the best model by AUC score
    # ─────────────────────────────────────────────
    best_result = max(results, key=lambda r: r["auc"])
    best_name = best_result["model"]
    best_model = trained_models[best_name]
    print(f"\n🏆 Best Model: {best_name} with AUC = {best_result['auc']:.4f}")

    # ─────────────────────────────────────────────
    # 8–10. Save the best model, scaler, and feature names
    # ─────────────────────────────────────────────
    os.makedirs(models_dir, exist_ok=True)

    joblib.dump(best_model,    os.path.join(models_dir, "churn_model.pkl"))
    joblib.dump(scaler,        os.path.join(models_dir, "scaler.pkl"))
    joblib.dump(feature_names, os.path.join(models_dir, "feature_names.pkl"))
    print(f"\nSaved: {models_dir}/churn_model.pkl")
    print(f"Saved: {models_dir}/scaler.pkl")
    print(f"Saved: {models_dir}/feature_names.pkl")

    # ─────────────────────────────────────────────
    # 11. Extract top 5 feature importances / coefficients
    #     Logistic Regression → coefficients (abs value)
    #     Decision Tree / Random Forest → feature_importances_
    # ─────────────────────────────────────────────
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    else:
        importances = np.abs(best_model.coef_[0])

    importance_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    top5 = importance_series.head(5).round(6).to_dict()

    top_features_path = os.path.join(models_dir, "top_features.json")
    with open(top_features_path, "w") as f:
        json.dump(top5, f, indent=4)
    print(f"Saved: {top_features_path}")

    # ─────────────────────────────────────────────
    # 12. Save all three model results to JSON
    # ─────────────────────────────────────────────
    model_results_path = os.path.join(models_dir, "model_results.json")
    with open(model_results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved: {model_results_path}")

    # ─────────────────────────────────────────────
    # Final summary line
    # ─────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"  ✅ WINNER: {best_name} | AUC = {best_result['auc']:.4f}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    train_and_evaluate()
