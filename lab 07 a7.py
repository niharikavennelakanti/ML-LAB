# -------------------------------------------------------------
# LAB 07 – Emotion Detection using Brain Activity Dataset
# Tasks: Hyperparameter Tuning, Model Comparison, XAI (SHAP, LIME)
# -------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Import ML models
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier

# Explainable AI libraries
import shap
import lime
import lime.lime_tabular

# -------------------------------------------------------------
# A1. Load Dataset
# -------------------------------------------------------------
data = pd.read_csv("data set 2.csv")
print("✅ Dataset Loaded Successfully")
print(data.head())

# Preprocess dataset
target_col = 'label' if 'label' in data.columns else data.columns[-1]
X = data.drop(columns=[target_col])
y = data[target_col]

# Encode target if categorical
if y.dtype == 'object':
    y = LabelEncoder().fit_transform(y)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# -------------------------------------------------------------
# A2. RandomizedSearchCV for Hyperparameter Tuning
# -------------------------------------------------------------
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10]
}

rf = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid_rf,
                                   n_iter=5, cv=3, verbose=2, n_jobs=-1)
random_search.fit(X_train, y_train)
print(f"\nBest Parameters (Random Forest): {random_search.best_params_}")

# -------------------------------------------------------------
# A3. Train and Compare Multiple Classifiers
# -------------------------------------------------------------
models = {
    "Perceptron": Perceptron(),
    "SVM": SVC(probability=True),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(**random_search.best_params_),
    "AdaBoost": AdaBoostClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
    "Naive Bayes": GaussianNB(),
    "MLP": MLPClassifier(max_iter=500),
    "CatBoost": CatBoostClassifier(verbose=0)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    results.append([name, train_acc, test_acc])
    print(f"\n{name} Model Results:")
    print(f"Train Accuracy: {train_acc:.3f}, Test Accuracy: {test_acc:.3f}")
    print(classification_report(y_test, y_pred_test))

# Tabulate results
results_df = pd.DataFrame(results, columns=['Model', 'Train Accuracy', 'Test Accuracy'])
print("\n📊 Model Comparison Table:")
print(results_df)

# Plot model comparison
plt.figure(figsize=(10,6))
sns.barplot(data=results_df, x='Model', y='Test Accuracy', palette='viridis')
plt.xticks(rotation=45)
plt.title("Model Accuracy Comparison (Lab 07)")
plt.show()

# -------------------------------------------------------------
# A5. Confusion Matrix for Best Model
# -------------------------------------------------------------
best_model_name = results_df.iloc[results_df['Test Accuracy'].idxmax()]['Model']
best_model = models[best_model_name]
y_pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print(f"🏆 Best Performing Model: {best_model_name}")

# -------------------------------------------------------------
# Optional O1: SHAP for Explainability
# -------------------------------------------------------------
explainer = shap.Explainer(best_model, X_train)
shap_values = explainer(X_test[:100])

# Feature importance summary
shap.summary_plot(shap_values, X_test, feature_names=X.columns)

# -------------------------------------------------------------
# Optional O2: LIME for Local Interpretability
# -------------------------------------------------------------
explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X.columns,
    class_names=[str(c) for c in np.unique(y)],
    mode='classification'
)

i = 10  # Random test sample
exp = explainer_lime.explain_instance(X_test[i], best_model.predict_proba)
exp.show_in_notebook(show_table=True)

print("\n✅ Lab 07 Execution Completed Successfully!")
