import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib

# ============================
# 1. Load dataset
# ============================
df = pd.read_csv("Obesity Classification.csv", sep=";")

TARGET = "label"
X = df.drop(columns=[TARGET])
y = df[TARGET]

# ============================
# 2. Encode categorical columns
# ============================
encoders = {}
categorical_cols = ["gender"]

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le  # Simpan encoder

# ============================
# 3. Train-test split
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================
# 4. SMOTE untuk balance class
# ============================
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# ============================
# 5. Train Random Forest
# ============================
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)
model.fit(X_train_res, y_train_res)

# ============================
# 6. Save model, encoder, feature list
# ============================
joblib.dump(model, "model_rf.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(list(X.columns), "feature_list.pkl")

print("Model, encoder, dan feature list berhasil disimpan!")
