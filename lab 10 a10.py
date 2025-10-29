# -------------------------------------------------------------
# LAB 10 – Emotion Detection using Brain Activity Dataset
# -------------------------------------------------------------
# Tasks:
# A1. Correlation Heatmap
# A2. PCA (99% variance)
# A3. PCA (95% variance)
# A4. Sequential Feature Selection
# A5. LIME and SHAP Explainability
# -------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SequentialFeatureSelector
import shap
import lime
import lime.lime_tabular

# -------------------------------------------------------------
# A1. Load Dataset and Correlation Heatmap
# -------------------------------------------------------------
def load_and_preprocess(filepath):
    """Load dataset, encode target, and standardize features."""
    data = pd.read_csv(filepath)
    target_col = 'label' if 'label' in data.columns else data.columns[-1]

    X = data.drop(columns=[target_col])
    y = data[target_col]

    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X, X_scaled, y, data.columns[:-1]


def plot_correlation_heatmap(X, feature_names):
    """Plot correlation heatmap for features."""
    plt.figure(figsize=(12, 8))
    corr = pd.DataFrame(X, columns=feature_names).corr()
    sns.heatmap(corr, cmap="coolwarm", annot=False)
    plt.title("Feature Correlation Heatmap (A1)")
    plt.show()


# -------------------------------------------------------------
# A2 & A3. PCA-based Dimensionality Reduction
# -------------------------------------------------------------
def pca_analysis(X_scaled, y, variance_ratio, model):
    """Apply PCA and train classifier at given variance retention."""
    pca = PCA(n_components=variance_ratio)
    X_pca = pca.fit_transform(X_scaled)
    print(f"\nVariance Retained: {np.sum(pca.explained_variance_ratio_):.4f}")

    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"✅ PCA ({variance_ratio*100:.0f}% variance): Accuracy = {acc:.3f}")
    return acc, pca


def plot_cumulative_variance(X_scaled):
    """Plot cumulative explained variance for PCA."""
    pca = PCA().fit(X_scaled)
    plt.figure(figsize=(8, 5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Cumulative PCA Variance (A2/A3)")
    plt.grid()
    plt.show()


# -------------------------------------------------------------
# A4. Sequential Feature Selection (SFS)
# -------------------------------------------------------------
def sequential_feature_selection(X_scaled, y, model, n_features=10):
    """Perform Sequential Forward Feature Selection."""
    sfs = SequentialFeatureSelector(model, n_features_to_select=n_features, direction='forward', cv=5, n_jobs=-1)
    sfs.fit(X_scaled, y)
    print(f"Selected Features Indices (A4): {sfs.get_support(indices=True)}")
    return sfs


# -------------------------------------------------------------
# A5. Explainable AI (LIME + SHAP)
# -------------------------------------------------------------
def explain_with_lime_shap(model, X_train, X_test, y_train, feature_names):
    """Explain predictions using LIME and SHAP."""
    # ---- LIME ----
    print("\n🔹 LIME Explainability:")
    explainer_lime = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=feature_names,
        class_names=[str(i) for i in np.unique(y_train)],
        mode='classification'
    )
    i = np.random.randint(0, len(X_test))
    exp = explainer_lime.explain_instance(X_test[i], model.predict_proba)
    exp.show_in_notebook(show_table=True)

    # ---- SHAP ----
    print("\n🔹 SHAP Explainability:")
    explainer_shap = shap.Explainer(model, X_train)
    shap_values = explainer_shap(X_test[:100])
    shap.summary_plot(shap_values, X_test, feature_names=feature_names)


# -------------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------------
if __name__ == "__main__":
    # A1. Correlation Heatmap
    X, X_scaled, y, feature_names = load_and_preprocess("data set 2.csv")
    plot_correlation_heatmap(X, feature_names)

    # A2 & A3. PCA 99% and 95% Variance
    base_model = RandomForestClassifier(n_estimators=100, random_state=42)
    plot_cumulative_variance(X_scaled)
    acc_99, pca_99 = pca_analysis(X_scaled, y, variance_ratio=0.99, model=base_model)
    acc_95, pca_95 = pca_analysis(X_scaled, y, variance_ratio=0.95, model=base_model)

    # A4. Sequential Feature Selection
    model = LogisticRegression(max_iter=500)
    sfs = sequential_feature_selection(X_scaled, y, model, n_features=10)

    # Evaluate with selected features
    X_sfs = X_scaled[:, sfs.get_support()]
    X_train, X_test, y_train, y_test = train_test_split(X_sfs, y, test_size=0.3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc_sfs = accuracy_score(y_test, y_pred)
    print(f"\n✅ Sequential Feature Selection Accuracy = {acc_sfs:.3f}")

    # A5. Explainability
    explain_with_lime_shap(model, X_train, X_test, y_train, np.array(feature_names)[sfs.get_support()])

    print("\n🎯 Lab 10 Completed Successfully!")
