import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
import os

# === Định nghĩa prefix & paths ===
prefix = "3D_CNN_pretrained"
drive_path = "/content/drive/MyDrive/MaxEnt"
embedding_dir = f"{drive_path}/Embedding"
model_dir = f"{drive_path}/models_baseline"
result_dir = f"{drive_path}/results_baseline"

# === Tạo thư mục nếu chưa có ===
os.makedirs(model_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)

# === Load datasets từ Drive ===
X_train = torch.load(f"{embedding_dir}/{prefix}_train_data.pt", weights_only=False)
y_train = torch.load(f"{embedding_dir}/{prefix}_train_labels.pt", weights_only=False)

X_val = torch.load(f"{embedding_dir}/{prefix}_val_data.pt", weights_only=False)
y_val = torch.load(f"{embedding_dir}/{prefix}_val_labels.pt", weights_only=False)

X_test = torch.load(f"{embedding_dir}/{prefix}_test_data.pt", weights_only=False)
y_test = torch.load(f"{embedding_dir}/{prefix}_test_labels.pt", weights_only=False)

# === Scale the data ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# === Train Logistic Regression (MaxEnt) ===
clf = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000)
clf.fit(X_train_scaled, y_train)

# === Evaluate on validation set ===
y_val_pred = clf.predict(X_val_scaled)
print(f"{prefix} Validation Classification Report:")
print(classification_report(y_val, y_val_pred))

# === Evaluate on test set ===
y_test_pred = clf.predict(X_test_scaled)
print(f"{prefix} Test Classification Report:")
print(classification_report(y_test, y_test_pred))

# === Save model and scaler to Drive ===
joblib.dump(clf, f"{model_dir}/maxent_{prefix}.joblib")
joblib.dump(scaler, f"{model_dir}/scaler_{prefix}.joblib")
print(f"✅ Saved model and scaler to '{model_dir}'")

# === Save report ===
with open(f"{result_dir}/report_maxent_{prefix}.txt", "w") as f:
    f.write("Validation Report:\n")
    f.write(classification_report(y_val, y_val_pred))
    f.write("\nTest Report:\n")
    f.write(classification_report(y_test, y_test_pred))
print(f"📝 Saved validation and test reports to '{result_dir}'")