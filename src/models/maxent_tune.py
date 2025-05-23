import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Categorical
import os
from scipy.stats import uniform
import joblib

# Định nghĩa paths và thư mục
prefix = "flatten"
drive_path = "/content/drive/MyDrive/MaxEnt"
embedding_dir = f"{drive_path}/Embedding"
result_dir = "maxent_tune_results"

os.makedirs(result_dir, exist_ok=True)

# Tải dữ liệu
def load_data(data_file, label_file):
    data = torch.load(data_file, weights_only=False)
    labels = torch.load(label_file, weights_only=False)
    return data, labels

train_data, train_labels = load_data(
    f"{embedding_dir}/{prefix}_train_data.pt",
    f"{embedding_dir}/{prefix}_train_labels.pt"
)
val_data, val_labels = load_data(
    f"{embedding_dir}/{prefix}_val_data.pt",
    f"{embedding_dir}/{prefix}_val_labels.pt"
)
test_data, test_labels = load_data(
    f"{embedding_dir}/{prefix}_test_data.pt",
    f"{embedding_dir}/{prefix}_test_labels.pt"
)

# Chuẩn hóa và giảm chiều
n_components = 500
scaler = StandardScaler()
pca = PCA(n_components=n_components)
X_train_scaled = scaler.fit_transform(train_data)
X_train_reduced = pca.fit_transform(X_train_scaled)
X_val_scaled = scaler.transform(val_data)
X_val_reduced = pca.transform(X_val_scaled)
X_test_scaled = scaler.transform(test_data)
X_test_reduced = pca.transform(X_test_scaled)

# Chuẩn hóa lại sau PCA
post_pca_scaler = StandardScaler()
X_train_reduced = post_pca_scaler.fit_transform(X_train_reduced)
X_val_reduced = post_pca_scaler.transform(X_val_reduced)
X_test_reduced = post_pca_scaler.transform(X_test_reduced)

# Kiểm tra dữ liệu
print("Train label distribution:", np.bincount(train_labels.astype(int)))
print("Variance of X_train_reduced:", np.var(X_train_reduced, axis=0).mean())

# Baseline MaxEnt
baseline_clf = LogisticRegression(max_iter=5000, solver='saga')
baseline_clf.fit(X_train_reduced, train_labels)
train_acc_baseline = accuracy_score(train_labels, baseline_clf.predict(X_train_reduced))
val_acc_baseline = accuracy_score(val_labels, baseline_clf.predict(X_val_reduced))
test_acc_baseline = accuracy_score(test_labels, baseline_clf.predict(X_test_reduced))
train_report_baseline = classification_report(train_labels, baseline_clf.predict(X_train_reduced))
val_report_baseline = classification_report(val_labels, baseline_clf.predict(X_val_reduced))
test_report_baseline = classification_report(test_labels, baseline_clf.predict(X_test_reduced))
print(f"Baseline MaxEnt Train Accuracy: {train_acc_baseline:.4f}")
print(f"Baseline MaxEnt Validation Accuracy: {val_acc_baseline:.4f}")
print(f"Baseline MaxEnt Test Accuracy: {test_acc_baseline:.4f}")
print(f"Baseline MaxEnt Train Classification Report:\n{train_report_baseline}")
print(f"Baseline MaxEnt Validation Classification Report:\n{val_report_baseline}")
print(f"Baseline MaxEnt Test Classification Report:\n{test_report_baseline}")

# Định nghĩa tham số cho tuning
param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'solver': ['lbfgs', 'saga', 'liblinear'],
    'class_weight': [None, 'balanced']
}

param_dist = {
    'C': uniform(0.1, 1000),
    'solver': ['lbfgs', 'saga', 'liblinear'],
    'class_weight': [None, 'balanced']
}

bayes_space = {
    'C': Real(0.1, 1000, prior='log-uniform'),
    'solver': Categorical(['lbfgs', 'saga', 'liblinear']),
    'class_weight': Categorical([None, 'balanced'])
}

# Grid Search
grid_search = GridSearchCV(LogisticRegression(max_iter=5000), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train_reduced, train_labels)
val_acc_grid = accuracy_score(val_labels, grid_search.predict(X_val_reduced))
print(f"Grid Search MaxEnt Validation Accuracy: {val_acc_grid:.4f}")
print(f"Best Grid Search Params: {grid_search.best_params_}")

# Random Search
random_search = RandomizedSearchCV(LogisticRegression(max_iter=5000), param_dist, n_iter=20, cv=5, n_jobs=-1)
random_search.fit(X_train_reduced, train_labels)
val_acc_random = accuracy_score(val_labels, random_search.predict(X_val_reduced))
print(f"Random Search MaxEnt Validation Accuracy: {val_acc_random:.4f}")
print(f"Best Random Search Params: {random_search.best_params_}")

# Bayesian Optimization
bayes_search = BayesSearchCV(LogisticRegression(max_iter=5000), bayes_space, n_iter=20, cv=5, n_jobs=-1)
bayes_search.fit(X_train_reduced, train_labels)
val_acc_bayes = accuracy_score(val_labels, bayes_search.predict(X_val_reduced))
print(f"Bayesian Optimization MaxEnt Validation Accuracy: {val_acc_bayes:.4f}")
print(f"Best Bayesian Optimization Params: {bayes_search.best_params_}")

# Chọn mô hình tốt nhất
tuning_results = [
    ('Grid Search', val_acc_grid, grid_search.best_estimator_),
    ('Random Search', val_acc_random, random_search.best_estimator_),
    ('Bayesian Optimization', val_acc_bayes, bayes_search.best_estimator_)
]
best_method, best_val_acc, best_model = max(tuning_results, key=lambda x: x[1])
print(f"Best Method: {best_method} with Validation Accuracy: {best_val_acc:.4f}")

# Đánh giá best model trên train, val, test
train_acc_best = accuracy_score(train_labels, best_model.predict(X_train_reduced))
test_acc_best = accuracy_score(test_labels, best_model.predict(X_test_reduced))
train_report_best = classification_report(train_labels, best_model.predict(X_train_reduced))
val_report_best = classification_report(val_labels, best_model.predict(X_val_reduced))
test_report_best = classification_report(test_labels, best_model.predict(X_test_reduced))
print(f"Best Model Train Accuracy: {train_acc_best:.4f}")
print(f"Best Model Validation Accuracy: {best_val_acc:.4f}")
print(f"Best Model Test Accuracy: {test_acc_best:.4f}")
print(f"Best Model Train Classification Report:\n{train_report_best}")
print(f"Best Model Validation Classification Report:\n{val_report_best}")
print(f"Best Model Test Classification Report:\n{test_report_best}")

# Lưu model tốt nhất
model_save_path = f"{result_dir}/best_maxent_model.joblib"
joblib.dump(best_model, model_save_path)
print(f"Best model saved to {model_save_path}")

# Lưu kết quả
with open(f"{result_dir}/maxent_flatten_tuning_results.txt", "w") as f:
    f.write(f"Baseline MaxEnt Train Accuracy: {train_acc_baseline:.4f}\n")
    f.write(f"Baseline MaxEnt Validation Accuracy: {val_acc_baseline:.4f}\n")
    f.write(f"Baseline MaxEnt Test Accuracy: {test_acc_baseline:.4f}\n")
    f.write(f"Baseline MaxEnt Train Classification Report:\n{train_report_baseline}\n")
    f.write(f"Baseline MaxEnt Validation Classification Report:\n{val_report_baseline}\n")
    f.write(f"Baseline MaxEnt Test Classification Report:\n{test_report_baseline}\n")
    f.write(f"Grid Search MaxEnt Validation Accuracy: {val_acc_grid:.4f}\n")
    f.write(f"Best Grid Search Params: {grid_search.best_params_}\n")
    f.write(f"Random Search MaxEnt Validation Accuracy: {val_acc_random:.4f}\n")
    f.write(f"Best Random Search Params: {random_search.best_params_}\n")
    f.write(f"Bayesian Optimization MaxEnt Validation Accuracy: {val_acc_bayes:.4f}\n")
    f.write(f"Best Bayesian Optimization Params: {bayes_search.best_params_}\n")
    f.write(f"Best Method: {best_method}\n")
    f.write(f"Best Model Train Accuracy: {train_acc_best:.4f}\n")
    f.write(f"Best Model Validation Accuracy: {best_val_acc:.4f}\n")
    f.write(f"Best Model Test Accuracy: {test_acc_best:.4f}\n")
    f.write(f"Best Model Train Classification Report:\n{train_report_best}\n")
    f.write(f"Best Model Validation Classification Report:\n{val_report_best}\n")
    f.write(f"Best Model Test Classification Report:\n{test_report_best}\n")
print(f"📝 Saved results to '{result_dir}/maxent_flatten_tuning_results.txt'")