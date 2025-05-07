# Machine Learning on Body Measurements using Scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from body_measurement_analysis import BodyMeasurements

# ---------------------------------------------
# Sample Dataset
# ---------------------------------------------
sample_data = [
    {"bust": "32GG", "waist": 28, "hips": 36},
    {"bust": "34D", "waist": 26, "hips": 37},
    {"bust": "36C", "waist": 29, "hips": 39},
    {"bust": "30E", "waist": 24, "hips": 35},
    {"bust": "38F", "waist": 32, "hips": 42},
    {"bust": "32C", "waist": 27, "hips": 36},
    {"bust": "36DD", "waist": 30, "hips": 40},
    {"bust": "34B", "waist": 25, "hips": 34},
    {"bust": "30D", "waist": 23, "hips": 33},
    {"bust": "40G", "waist": 35, "hips": 44}
]

# ---------------------------------------------
# Feature Extraction
# ---------------------------------------------
def extract_features(data):
    features, labels = [], []
    for entry in data:
        person = BodyMeasurements(bust=entry["bust"], waist=entry["waist"], hips=entry["hips"])
        features.append([
            person.bust_measurement(),
            person.waist,
            person.hips,
            person.waist_to_hip_ratio(),
            person.bust_to_waist_ratio(),
            person.hips_to_waist_ratio(),
            person.symmetry_score(),
            person.proportion_score(),
            person.curve_ratio(),
            person.size_consistency()
        ])
        labels.append(person.body_type())
    df = pd.DataFrame(features, columns=[
        "Bust (in)", "Waist (in)", "Hips (in)", "WHR", "BWR", "HWR",
        "Symmetry", "Proportion", "Curve", "Size Consistency"
    ])
    return df, pd.Series(labels)

# ---------------------------------------------
# Model Evaluation
# ---------------------------------------------
def evaluate_model(model, X_train, X_test, y_train, y_test, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n=== {name} ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# ---------------------------------------------
# Load and Preprocess Data
# ---------------------------------------------
X, y = extract_features(sample_data)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------
# Model Training and Evaluation
# ---------------------------------------------
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=200),
    "SVM": SVC(kernel='rbf', probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=3)
}

for name, model in models.items():
    evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test, name)

# ---------------------------------------------
# Feature Importance (Random Forest)
# ---------------------------------------------
def plot_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        sorted_idx = importances.argsort()
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(importances)), importances[sorted_idx], align='center')
        plt.yticks(range(len(importances)), [feature_names[i] for i in sorted_idx])
        plt.title("Feature Importance")
        plt.xlabel("Relative Importance")
        plt.tight_layout()
        plt.show()

plot_feature_importance(models["Random Forest"], X.columns)

# ---------------------------------------------
# PCA Visualization
# ---------------------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap='viridis', edgecolor='k')
plt.title("PCA Projection")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()

# ---------------------------------------------
# Permutation Importance
# ---------------------------------------------
perm = permutation_importance(models["Random Forest"], X_test_scaled, y_test, n_repeats=10, random_state=42)
sorted_idx = perm.importances_mean.argsort()
plt.figure(figsize=(10, 6))
plt.boxplot(perm.importances[sorted_idx].T, vert=False, labels=X.columns[sorted_idx])
plt.title("Permutation Importances (Test Set)")
plt.tight_layout()
plt.show()

# ---------------------------------------------
# Hyperparameter Tuning with GridSearchCV
# ---------------------------------------------
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5]
}
gs = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
gs.fit(X_train_scaled, y_train)
print("\nBest Parameters from Grid Search:", gs.best_params_)
print("Best Cross-Validation Score:", gs.best_score_)
