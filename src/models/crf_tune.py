import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import pycrfsuite
from collections import Counter
import scipy.spatial.distance as distance
from itertools import product
import os

# === Cấu hình ===
prefix = "flatten"
drive_path = "/content/drive/MyDrive/MaxEnt"
embedding_dir = f"{drive_path}/Embedding"
model_dir = "crf_models"
result_dir = "crf_results"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)

# Kiểm tra GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

# Kiểm tra kích thước dữ liệu
print(f"Train data shape: {train_data.shape}, Memory: {train_data.nbytes / 1024**2:.2f} MB")
print("Train label distribution:", np.bincount(train_labels.astype(int)))

# Chuẩn hóa và giảm chiều
n_components = 300
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_data)
X_val_scaled = scaler.transform(val_data)
X_test_scaled = scaler.transform(test_data)

pca = PCA(n_components=n_components)
X_train_reduced = pca.fit_transform(X_train_scaled)
X_val_reduced = pca.transform(X_val_scaled)
X_test_reduced = pca.transform(X_test_scaled)

# In tỷ lệ phương sai
print(f"PCA variance explained: {np.sum(pca.explained_variance_ratio_)*100:.2f}%")
print(f"X_train_reduced shape: {X_train_reduced.shape}, Memory: {X_train_reduced.nbytes / 1024**2:.2f} MB")

# MLP cho unary potentials
class UnaryMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(UnaryMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.layers(x)

def train_mlp(X_train, y_train, X_val, y_val, input_dim, hidden_dim=100, output_dim=None):
    if output_dim is None:
        output_dim = len(np.unique(y_train))
    
    model = UnaryMLP(input_dim, hidden_dim, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)
    
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    best_val_acc = 0
    best_model_state = None
    
    for epoch in range(30):
        model.train()
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_preds = torch.argmax(val_outputs, dim=1)
            val_acc = accuracy_score(y_val, val_preds.cpu().numpy())
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Val Accuracy: {val_acc:.4f}")
    
    model.load_state_dict(best_model_state)
    return model, best_val_acc

def predict_proba_mlp(model, X):
    model.eval()
    X_tensor = torch.FloatTensor(X).to(device)
    with torch.no_grad():
        probs = model(X_tensor).cpu().numpy()
    return probs

mlp_model, mlp_val_acc = train_mlp(X_train_reduced, train_labels, X_val_reduced, val_labels, input_dim=n_components)
print(f"MLP Validation Accuracy: {mlp_val_acc:.4f}")

# Tạo đặc trưng CRF
def extract_crf_features(X, unary_model, num_regions=20):
    X_crf = []
    for sample in X:
        features = []
        regions = np.array_split(sample, num_regions)
        unary_probs = predict_proba_mlp(unary_model, sample.reshape(1, -1))[0]
        for i, region in enumerate(regions):
            feat_dict = {
                f'region_{i}_mean': str(np.mean(region)),
                f'region_{i}_std': str(np.std(region)),
                f'region_{i}_min': str(np.min(region)),
                f'region_{i}_max': str(np.max(region)),
            }
            for cls, prob in enumerate(unary_probs):
                feat_dict[f'unary_{cls}'] = str(prob)
            for j, other_region in enumerate(regions):
                if i != j:
                    dist = distance.euclidean(region, other_region)
                    feat_dict[f'dist_to_region_{j}'] = str(dist)
                    feat_dict[f'cosine_to_region_{j}'] = str(1 - distance.cosine(region, other_region))
            features.append(feat_dict)
        X_crf.append(features)
    return X_crf

X_train_crf = extract_crf_features(X_train_reduced, mlp_model)
X_val_crf = extract_crf_features(X_val_reduced, mlp_model)
X_test_crf = extract_crf_features(X_test_reduced, mlp_model)
y_train_crf = [[str(int(label))] * 20 for label in train_labels]
y_val_crf = [[str(int(label))] * 20 for label in val_labels]
y_test_crf = [[str(int(label))] * 20 for label in test_labels]

# --- Grid Search ---
param_grid = {
    'c1': [0.1, 0.5],
    'c2': [0.1, 0.5],
    'max_iterations': [100]
}
param_list_grid = [
    {'c1': c1, 'c2': c2, 'max_iterations': mi}
    for c1, c2, mi in product(param_grid['c1'], param_grid['c2'], param_grid['max_iterations'])
]

def train_and_evaluate_crf(params, model_path_suffix):
    trainer = pycrfsuite.Trainer(verbose=False)
    for x_seq, y_seq in zip(X_train_crf, y_train_crf):
        trainer.append(x_seq, y_seq)
    trainer.set_params({
        'c1': params['c1'],
        'c2': params['c2'],
        'max_iterations': params['max_iterations'],
        'feature.possible_transitions': True
    })
    model_path = f"{model_dir}/crf_model_{model_path_suffix}.model"
    trainer.train(model_path)

    y_val_pred = predict_global_labels(X_val_crf, model_path)
    val_acc = accuracy_score(val_labels, y_val_pred)
    return val_acc, model_path, params

# Grid Search tuning
best_val_acc_grid = 0
best_params_grid = None
best_model_path_grid = None
for idx, params in enumerate(param_list_grid):
    val_acc, model_path, params_used = train_and_evaluate_crf(params, f"grid_{idx}")
    print(f"Grid Search - Params: {params_used}, Val Accuracy: {val_acc:.4f}")
    if val_acc > best_val_acc_grid:
        best_val_acc_grid = val_acc
        best_params_grid = params_used
        best_model_path_grid = model_path

# --- Random Search ---
param_grid_random = {
    'c1': [0.01, 0.1, 0.3, 0.5, 0.7, 1.0],
    'c2': [0.01, 0.1, 0.3, 0.5, 0.7, 1.0],
    'max_iterations': [100]
}
n_random_trials = 10

best_val_acc_random = 0
best_params_random = None
best_model_path_random = None

for i in range(n_random_trials):
    params = {
        'c1': random.choice(param_grid_random['c1']),
        'c2': random.choice(param_grid_random['c2']),
        'max_iterations': 100
    }
    val_acc, model_path, params_used = train_and_evaluate_crf(params, f"random_{i}")
    print(f"Random Search - Trial {i}, Params: {params_used}, Val Accuracy: {val_acc:.4f}")
    if val_acc > best_val_acc_random:
        best_val_acc_random = val_acc
        best_params_random = params_used
        best_model_path_random = model_path

# --- So sánh Grid và Random Search ---
if best_val_acc_grid >= best_val_acc_random:
    best_val_acc = best_val_acc_grid
    best_params = best_params_grid
    best_model_path = best_model_path_grid
    print(f"Chọn Grid Search với Validation Accuracy: {best_val_acc:.4f}")
else:
    best_val_acc = best_val_acc_random
    best_params = best_params_random
    best_model_path = best_model_path_random
    print(f"Chọn Random Search với Validation Accuracy: {best_val_acc:.4f}")

# Đánh giá trên test set
y_test_pred = predict_global_labels(X_test_crf, best_model_path)
test_acc_best = accuracy_score(test_labels, y_test_pred)

print(f"Best CRF Test Accuracy: {test_acc_best:.4f}")
print(f"Best Params: {best_params}")

# Lưu kết quả
with open(f"{result_dir}/crf_flatten_tuning_comparison_results.txt", "w") as f:
    f.write(f"Best Grid Search Validation Accuracy: {best_val_acc_grid:.4f}\n")
    f.write(f"Best Grid Search Params: {best_params_grid}\n")
    f.write(f"Best Random Search Validation Accuracy: {best_val_acc_random:.4f}\n")
    f.write(f"Best Random Search Params: {best_params_random}\n")
    f.write(f"Final Selected Validation Accuracy: {best_val_acc:.4f}\n")
    f.write(f"Final Selected Test Accuracy: {test_acc_best:.4f}\n")
    f.write(f"Final Selected Params: {best_params}\n")