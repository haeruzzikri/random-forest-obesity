import pandas as pd
from sklearn.model_selection import train_test_split
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
# 2. Identify categorical and numerical columns
# ============================
categorical_cols = [
    'Gender', 'CALC', 'FAVC', 'SCC', 'SMOKE',
    'family_history_with_overweight', 'CAEC', 'MTRANS'
]

label_col = 'NObeyesdad'

numeric_cols = [
    'Age', 'Height', 'Weight', 'FCVC', 'NCP',
    'CH2O', 'FAF', 'TUE'
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
# 8. Evaluation
# ============================
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
cm = confusion_matrix(y_test, y_pred)

print("===== Evaluation Result =====")
print(f"Akurasi: {acc:.4f}")
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
fi = model.feature_importances_
fi_df = pd.DataFrame({
    "Feature": feature_list,
    "Importance": fi
}).sort_values("Importance", ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(
    x="Importance",
    y="Feature",
    data=fi_df,
    hue="Feature",
    dodge=False,
    palette="viridis",
    legend=False
)
plt.title("Feature Importance - Random Forest")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()


# ============================
# 11. Save Classification Report as CSV
# ============================
pd.DataFrame(report).transpose().to_csv("classification_report.csv")

# ============================
# 12. Save Metric Summary as TXT
# ============================
with open("metrics_summary.txt", "w") as f:
    f.write("=== METRIC SUMMARY ===\n")
    f.write(f"Akurasi: {acc:.4f}\n\n")
    f.write(classification_report(y_test, y_pred))

# ============================
# 13. Save Model + Encoders + Feature List
# ============================
joblib.dump(model, "model_rf.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(feature_list, "feature_list.pkl")

print("\nTraining selesai! Semua file model + evaluasi tersimpan.")
