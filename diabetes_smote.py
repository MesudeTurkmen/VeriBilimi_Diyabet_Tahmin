import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, ConfusionMatrixDisplay
)
from imblearn.over_sampling import SMOTE

# Modeller
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

# === Klasör oluştur ===
os.makedirs("outputs", exist_ok=True)

# === Veri Hazırlama ===
df = pd.read_csv("diabetes_binary_health_indicators_BRFSS2015.csv")
df = df[
    (df['BMI'].between(12, 60)) &
    (df['GenHlth'].between(1, 5)) &
    (df['PhysHlth'].between(0, 30)) &
    (df['MentHlth'].between(0, 30))
].drop_duplicates()

X = df.drop("Diabetes_binary", axis=1)
y = df["Diabetes_binary"]

# SMOTE
X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)

# Split ve ölçekleme
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Modeller ===
models = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1000),
        "params": {'C': [0.1, 1], 'solver': ['lbfgs']}
    },
    "Decision Tree": {
        "model": DecisionTreeClassifier(),
        "params": {'max_depth': [5, 10], 'min_samples_split': [2, 5]}
    },
    "Random Forest": {
        "model": RandomForestClassifier(),
        "params": {'n_estimators': [100], 'max_depth': [10]}
    },
    "K-NN": {
        "model": KNeighborsClassifier(),
        "params": {'n_neighbors': [5], 'weights': ['uniform']}
    },
    "Naive Bayes": {
        "model": GaussianNB(),
        "params": {}
    },
    "Gradient Boosting": {
        "model": GradientBoostingClassifier(),
        "params": {'n_estimators': [100], 'learning_rate': [0.1]}
    },
    "XGBoost": {
        "model": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "params": {'n_estimators': [100], 'max_depth': [3]}
    }
}

# === Eğitim, değerlendirme ve grafikler ===
results = []

for name, config in models.items():
    print(f"🔍 {name} eğitiliyor...")
    grid = GridSearchCV(config["model"], config["params"], cv=3, scoring='roc_auc', n_jobs=-1)

    if name in ["Logistic Regression", "K-NN"]:
        grid.fit(X_train_scaled, y_train)
        y_proba = grid.predict_proba(X_test_scaled)[:, 1]
        y_pred = grid.predict(X_test_scaled)
    else:
        grid.fit(X_train, y_train)
        y_proba = grid.predict_proba(X_test)[:, 1]
        y_pred = grid.predict(X_test)

    # ROC Eğrisi
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_proba):.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{name} - ROC Curve")
    plt.legend()
    plt.savefig(f"outputs/roc_{name.replace(' ', '_')}.png")
    plt.close()

    # Confusion Matrix
    plt.figure()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["No Diabetes", "Diabetes"])
    plt.title(f"{name} - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"outputs/cm_{name.replace(' ', '_')}.png")
    plt.close()

    # Feature Importance
    if hasattr(grid.best_estimator_, "feature_importances_"):
        importances = grid.best_estimator_.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        plt.figure(figsize=(8, 6))
        sns.barplot(x=importances[sorted_idx], y=X.columns[sorted_idx])
        plt.title(f"{name} - Feature Importance")
        plt.tight_layout()
        plt.savefig(f"outputs/fi_{name.replace(' ', '_')}.png")
        plt.close()

    results.append({
        "Model": name,
        "Best Params": grid.best_params_,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba)
    })

# === Sonuçları Yazdır ===
df_results = pd.DataFrame(results)
df_results = df_results.sort_values(by="ROC-AUC", ascending=False)
df_results.to_csv("outputs/model_sonuclari.csv", index=False)
print("\n✅ Tüm grafikler outputs/ klasörüne kaydedildi.")
print(df_results)
