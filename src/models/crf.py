import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
import pycrfsuite
from collections import Counter
import scipy.spatial.distance as distance
import os

# === Cấu hình ===
prefix = "flatten"
drive_path = "/content/drive/MyDrive/MaxEnt"
embedding_dir = f"{drive_path}/Embedding"
model_dir = "crf_models_base"
result_dir = "crf_results_base"
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
n_components = 500
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

    for epoch in range(50):
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
def extract_crf_features(X, unary_model, num_regions=25):
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
y_train_crf = [[str(int(label))] * 25 for label in train_labels]
y_val_crf = [[str(int(label))] * 25 for label in val_labels]
y_test_crf = [[str(int(label))] * 25 for label in test_labels]

# Huấn luyện CRF baseline
trainer = pycrfsuite.Trainer(verbose=False)
for x_seq, y_seq in zip(X_train_crf, y_train_crf):
    trainer.append(x_seq, y_seq)
trainer.set_params({
    'c1': 0.1,
    'c2': 0.1,
    'max_iterations': 100,
    'feature.possible_transitions': True
})
model_path = f"{model_dir}/medmnist_flatten_crf.model"
trainer.train(model_path)

# Dự đoán CRF
def predict_global_labels(X_crf, model_file):
    tagger = pycrfsuite.Tagger()
    tagger.open(model_file)
    y_pred_global = []
    for x_seq in X_crf:
        pred = tagger.tag(x_seq)
        counts = Counter(pred)
        most_common_label = counts.most_common(1)[0][0]
        y_pred_global.append(int(most_common_label))
    return np.array(y_pred_global)

y_train_pred = predict_global_labels(X_train_crf, model_path)
y_val_pred = predict_global_labels(X_val_crf, model_path)
y_test_pred = predict_global_labels(X_test_crf, model_path)

# Đánh giá và lưu kết quả
train_accuracy = accuracy_score(train_labels, y_train_pred)
val_accuracy = accuracy_score(val_labels, y_val_pred)
test_accuracy = accuracy_score(test_labels, y_test_pred)
print(f"CRF Train Accuracy: {train_accuracy:.4f}")
print(f"CRF Validation Accuracy: {val_accuracy:.4f}")
print(f"CRF Test Accuracy: {test_accuracy:.4f}")

# Classification report
train_report = classification_report(train_labels, y_train_pred)
val_report = classification_report(val_labels, y_val_pred)
test_report = classification_report(test_labels, y_test_pred)
print(f"CRF Train Classification Report:\n{train_report}")
print(f"CRF Validation Classification Report:\n{val_report}")
print(f"CRF Test Classification Report:\n{test_report}")

with open(f"{result_dir}/crf_{prefix}_baseline_results.txt", "w") as f:
    f.write(f"MLP Validation Accuracy: {mlp_val_acc:.4f}\n")
    f.write(f"CRF Train Accuracy: {train_accuracy:.4f}\n")
    f.write(f"CRF Validation Accuracy: {val_accuracy:.4f}\n")
    f.write(f"CRF Test Accuracy: {test_accuracy:.4f}\n")
    f.write("\nCRF Train Classification Report:\n")
    f.write(train_report)
    f.write("\nCRF Validation Classification Report:\n")
    f.write(val_report)
    f.write("\nCRF Test Classification Report:\n")
    f.write(test_report)
print(f"📝 Saved results to '{result_dir}/crf_{prefix}_baseline_results.txt'")