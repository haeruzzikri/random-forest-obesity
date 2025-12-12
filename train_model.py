import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from imblearn.over_sampling import SMOTE
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# ============================
# 1. Load Dataset
# ============================
df = pd.read_csv("Obesity_Data_Set.csv")

# ============================
# 2. Missing Value Handling (NEW)
# ============================
# Numeric → fill median
df.fillna(df.median(numeric_only=True), inplace=True)

# Categorical → fill mode
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].fillna(df[col].mode()[0])


# ============================
# 3. Add BMI Column (NEW)
# ============================
df["BMI"] = df["Weight"] / (df["Height"] ** 2)

# ============================
# 2. Identify categorical and numerical columns
# ============================
categorical_cols = [
    'Gender', 'CALC', 'FAVC', 'SCC', 'SMOKE',
    'family_history_with_overweight', 'CAEC', 'MTRANS'
]

label_col = 'NObeyesdad'

numeric_cols = [
    'Age', 'Height', 'Weight', 'FCVC', 'NCP',
    'CH2O', 'FAF', 'TUE', 'BMI'  # BMI ditambahkan
]

# ============================
# 3. Encode categorical columns
# ============================
encoders = {}
for col in categorical_cols + [label_col]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ============================
# 4. Prepare features and labels
# ============================
feature_list = numeric_cols + categorical_cols
X = df[feature_list]
y = df[label_col]

# ============================
# 5. Train-test split
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================
# 6. Apply SMOTE
# ============================
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

# ============================
# 7. Train Random Forest
# ============================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42
)
model.fit(X_train_sm, y_train_sm)

# ============================
# 8. Evaluation (updated)
# ============================
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
report_dict = classification_report(y_test, y_pred, output_dict=True)
cm = confusion_matrix(y_test, y_pred)

# Ambil metrik macro avg
macro_p = report_dict["macro avg"]["precision"]
macro_r = report_dict["macro avg"]["recall"]
macro_f1 = report_dict["macro avg"]["f1-score"]

# ============================
# 11. Cross Validation (NEW)
# ============================
cv_scores = cross_val_score(model, X, y, cv=5)
cv_mean = cv_scores.mean()

print("===== Evaluation Result =====")
print(f"Akurasi: {acc:.4f}")
print(f"Precision (macro): {macro_p:.4f}")
print(f"Recall (macro): {macro_r:.4f}")
print(f"F1-Score (macro): {macro_f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(cm)


# ============================
# 9. Save Confusion Matrix Plot
# ============================
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix.png")
plt.close()

# ============================
# 10. Save Feature Importance
# ============================
fi_df = pd.DataFrame({
    "Feature": feature_list,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(
    x="Importance",
    y="Feature",
    data=fi_df,
    dodge=False,
    legend=False
)
plt.title("Feature Importance - Random Forest")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

# ============================
# 11. Save Evaluation CSV
# ============================
pd.DataFrame(report_dict).transpose().to_csv("classification_report.csv")

# ============================
# 12. Save Metric Summary
# ============================
with open("metrics_summary.txt", "w") as f:
    f.write("=== METRIC SUMMARY ===\n")
    f.write(f"Akurasi: {acc:.4f}\n\n")
    f.write(classification_report(y_test, y_pred))

# ============================
# 13. Save Evaluation Object for Streamlit
# ============================

# Ambil weighted average metrics dari classification_report
precision = report_dict["weighted avg"]["precision"]
recall = report_dict["weighted avg"]["recall"]
f1 = report_dict["weighted avg"]["f1-score"]

evaluation = {
    "accuracy": acc,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "classification_report": report_dict,
    "confusion_matrix": cm,
    "feature_importance": fi_df,
    "class_names": encoders[label_col].classes_
}
joblib.dump(evaluation, "evaluation.pkl")

# ============================
# 14. Save Model + Encoders + Feature List
# ============================
joblib.dump(model, "model_rf.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(feature_list, "feature_list.pkl")

print("\nTraining selesai! Semua file model + evaluasi tersimpan.")
