# -------------------------------------------------------------
# LAB 09 – Emotion Detection using Brain Activity Dataset
# -------------------------------------------------------------
# Topics Covered:
# A1. Stacking Classifier
# A2. Pipeline Construction
# A3. LIME Explainability
# -------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import lime
import lime.lime_tabular

# -------------------------------------------------------------
# A1. Load Dataset and Preprocessing
# -------------------------------------------------------------
def load_and_preprocess_dataset(filepath):
    """Load dataset, encode labels, and standardize features"""
    data = pd.read_csv(filepath)
    target_col = 'label' if 'label' in data.columns else data.columns[-1]

    X = data.drop(columns=[target_col])
    y = data[target_col]

    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test, X.columns

# -------------------------------------------------------------
# A2. Build Stacking Classifier
# -------------------------------------------------------------
def build_stacking_classifier():
    """Define base and meta models for stacking ensemble"""
    base_models = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('svm', SVC(probability=True, kernel='rbf'))
    ]

    meta_model = LogisticRegression(max_iter=500)
    stacking_clf = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)
    return stacking_clf

# -------------------------------------------------------------
# A3. Build a Complete Pipeline
# -------------------------------------------------------------
def build_pipeline(model):
    """Construct a pipeline for preprocessing + classification"""
    pipe = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])
    return pipe

# -------------------------------------------------------------
# A4. Train, Evaluate, and Visualize
# -------------------------------------------------------------
def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Train and evaluate model performance"""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Model Accuracy: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='g', cmap='coolwarm')
    plt.title("Confusion Matrix - Stacking Classifier")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    return acc

# -------------------------------------------------------------
# A5. Explain Model Predictions using LIME
# -------------------------------------------------------------
def explain_with_lime(model, X_train, X_test, feature_names, class_names):
    """Use LIME to explain individual predictions"""
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=feature_names,
        class_names=[str(c) for c in np.unique(class_names)],
        mode='classification'
    )

    i = np.random.randint(0, len(X_test))
    exp = explainer.explain_instance(X_test[i], model.predict_proba)
    exp.show_in_notebook(show_table=True)
    print(f"\nExplained instance index: {i}")

# -------------------------------------------------------------
# MAIN EXECUTION (All print statements here)
# -------------------------------------------------------------
if __name__ == "__main__":
    print("🔹 Loading and Preprocessing Data...")
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_dataset("data set 2.csv")

    print("🔹 Building Stacking Classifier...")
    stacking_model = build_stacking_classifier()

    print("🔹 Creating Pipeline...")
    pipeline = build_pipeline(stacking_model)

    print("🔹 Training and Evaluating Model...")
    accuracy = evaluate_model(pipeline, X_train, X_test, y_train, y_test)
    print(f"Final Stacking Pipeline Accuracy: {accuracy:.3f}")

    print("🔹 Running LIME Explainability...")
    explain_with_lime(pipeline, X_train, X_test, feature_names, y_train)

    print("\n✅ Lab 09 Execution Completed Successfully!")
