import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, roc_curve
)

# Load cleaned dataset
df = pd.read_csv("cleaned_telco_final.csv")

# Separate features and target
X = df.drop(columns=['Churn_Yes'])
y = df['Churn_Yes']

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize models
logreg = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train models
logreg.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Predict
y_pred_logreg = logreg.predict(X_test)
y_pred_rf = rf.predict(X_test)

# Probabilities for ROC
y_prob_logreg = logreg.predict_proba(X_test)[:, 1]
y_prob_rf = rf.predict_proba(X_test)[:, 1]

# Evaluation function
def evaluate_model(name, y_true, y_pred, y_prob):
    print(f"\n📊 {name} Performance:")
    print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
    print("Precision:", round(precision_score(y_true, y_pred), 4))
    print("Recall:", round(recall_score(y_true, y_pred), 4))
    print("F1 Score:", round(f1_score(y_true, y_pred), 4))
    print("ROC AUC:", round(roc_auc_score(y_true, y_prob), 4))
    print("\nClassification Report:\n", classification_report(y_true, y_pred))

# Evaluate both models
evaluate_model("Logistic Regression", y_test, y_pred_logreg, y_prob_logreg)
evaluate_model("Random Forest", y_test, y_pred_rf, y_prob_rf)

# ROC Curve comparison
fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_logreg)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

plt.figure(figsize=(8, 6))
plt.plot(fpr_log, tpr_log, label='Logistic Regression', linestyle='--')
plt.plot(fpr_rf, tpr_rf, label='Random Forest')
plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve_comparison.png")
plt.close()

print("\n✅ Modeling complete. ROC curve saved as 'roc_curve_comparison.png'.")
