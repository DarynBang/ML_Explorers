import numpy as np
import torch
from src.data.make_dataset import Custom2DDataset, Custom3DDataset
from src.models.CNN2D import CNN2D_MLP
from src.models.CNN3D import CNN3D_MLP
from torch.nn.utils import DataLoader


flatten_train_data = np.reshape(train_data, (train_data.shape[0], -1))
flatten_val_data = np.reshape(val_data, (val_data.shape[0], -1))
flatten_test_data = np.reshape(test_data, (test_data.shape[0], -1))
# print(flatten_train_data.shape)
# print(flatten_val_data.shape)
# print(flatten_test_data.shape)
# print(train_labels.shape)
# print(val_labels.shape)
# print(test_labels.shape)

# Save flatten data
torch.save(flatten_train_data,  "Embedding/flatten_train_data.pt")
torch.save(flatten_val_data,    "Embedding/flatten_val_data.pt")
torch.save(flatten_test_data,   "Embedding/flatten_test_data.pt")
torch.save(train_labels,        "Embedding/flatten_train_labels.pt")
torch.save(val_labels,          "Embedding/flatten_val_labels.pt")
torch.save(test_labels,         "Embedding/flatten_test_labels.pt")


train2D_dataset = Custom2DDataset(train_data, train_labels)
val2D_dataset = Custom2DDataset(val_data, val_labels)
test2D_dataset = Custom2DDataset(test_data, test_labels)

train2D_loader = DataLoader(train2D_dataset, batch_size=512, shuffle=False)
val2D_loader = DataLoader(val2D_dataset, batch_size=512, shuffle=False)
test2D_loader = DataLoader(test2D_dataset, batch_size=512, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
model_2D = CNN2D_MLP(num_classes=6).to(device)
model_2D.eval()

vectors2D_train = []
vectors2D_val = []
vectors2D_test = []

# Get train embeddings from 2D model
with torch.no_grad():
    for inputs, labels in train2D_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        vectors2D_train.append(model_2D.get_embedding(inputs))
    vectors2D_train = torch.concatenate(vectors2D_train).detach().cpu().numpy()
    print(f'Train Embeddings after 2D Model: {vectors2D_train.shape}')

# Get validation embeddings from 2D model
with torch.no_grad():
    for inputs, labels in val2D_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        vectors2D_val.append(model_2D.get_embedding(inputs))
    vectors2D_val = torch.concatenate(vectors2D_val).detach().cpu().numpy()
    print(f'Val Embeddings after 2D Model: {vectors2D_val.shape}')

# Get test embeddings from 2D model
with torch.no_grad():
    for inputs, labels in test2D_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        vectors2D_test.append(model_2D.get_embedding(inputs))
    vectors2D_test = torch.concatenate(vectors2D_test).detach().cpu().numpy()
    print(f'Test Embeddings after 2D Model: {vectors2D_test.shape}')

# Save initial 2D data
torch.save(vectors2D_train,     "Embedding/2D_CNN_init_train_data.pt")
torch.save(vectors2D_val,       "Embedding/2D_CNN_init_val_data.pt")
torch.save(vectors2D_test,      "Embedding/2D_CNN_init_test_data.pt")
torch.save(train_labels,        "Embedding/2D_CNN_init_train_labels.pt")
torch.save(val_labels,          "Embedding/2D_CNN_init_val_labels.pt")
torch.save(test_labels,         "Embedding/2D_CNN_init_test_labels.pt")


# Load states of trained models
model_2D = CNN2D_MLP(num_classes=6).to(device)
model_2D.load_state_dict(torch.load("best_model_2D_CNN.pth"))
model_2D.eval()


vectors2D_train = []
vectors2D_val = []
vectors2D_test = []

# Get train embeddings from 2D model
with torch.no_grad():
    for inputs, labels in train2D_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        vectors2D_train.append(model_2D.get_embedding(inputs))
    vectors2D_train = torch.concatenate(vectors2D_train).detach().cpu().numpy()
    print(f'Train Embeddings after 2D Model: {vectors2D_train.shape}')

# Get validation embeddings from 2D model
with torch.no_grad():
    for inputs, labels in val2D_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        vectors2D_val.append(model_2D.get_embedding(inputs))
    vectors2D_val = torch.concatenate(vectors2D_val).detach().cpu().numpy()
    print(f'Val Embeddings after 2D Model: {vectors2D_val.shape}')

# Get test embeddings from 2D model
with torch.no_grad():
    for inputs, labels in test2D_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        vectors2D_test.append(model_2D.get_embedding(inputs))
    vectors2D_test = torch.concatenate(vectors2D_test).detach().cpu().numpy()
    print(f'Test Embeddings after 2D Model: {vectors2D_test.shape}')

# Save trained embeddings from 2D model
torch.save(vectors2D_train,     "Embedding/2D_CNN_pretrained_train_data.pt")
torch.save(vectors2D_val,       "Embedding/2D_CNN_pretrained_val_data.pt")
torch.save(vectors2D_test,      "Embedding/2D_CNN_pretrained_test_data.pt")
torch.save(train_labels,        "Embedding/2D_CNN_pretrained_train_labels.pt")
torch.save(val_labels,          "Embedding/2D_CNN_pretrained_val_labels.pt")
torch.save(test_labels,         "Embedding/2D_CNN_pretrained_test_labels.pt")


train3D_dataset = Custom3DDataset(train_data, train_labels)
val3D_dataset = Custom3DDataset(val_data, val_labels)
test3D_dataset = Custom3DDataset(test_data, test_labels)

train3D_loader = DataLoader(train3D_dataset, batch_size=512, shuffle=False)
val3D_loader = DataLoader(val3D_dataset, batch_size=512, shuffle=False)
test3D_loader = DataLoader(test3D_dataset, batch_size=512, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
model_3D = CNN3D_MLP(num_classes=6).to(device)
model_3D.eval()

vectors3D_train = []
vectors3D_val = []
vectors3D_test = []

# Get Train Embeddings from 3D model
with torch.no_grad():
    for inputs, labels in train3D_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        vectors3D_train.append(model_3D.embedding(inputs))
    vectors3D_train = torch.concatenate(vectors3D_train).detach().cpu().numpy()
    print(f'Train Embeddings after 3D Model: {vectors3D_train.shape}')

# Get Validation Embeddings from 3D model
with torch.no_grad():
    for inputs, labels in val3D_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        vectors3D_val.append(model_3D.embedding(inputs))
    vectors3D_val = torch.concatenate(vectors3D_val).detach().cpu().numpy()
    print(f'Val Embeddings after 3D Model: {vectors3D_val.shape}')

# Get Test Embeddings from 2D model
with torch.no_grad():
    for inputs, labels in test3D_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        vectors3D_test.append(model_3D.embedding(inputs))
    vectors3D_test = torch.concatenate(vectors3D_test).detach().cpu().numpy()
    print(f'Test Embeddings after 3D Model: {vectors3D_test.shape}')

# Save initial 3D data
torch.save(vectors3D_train,     "Embedding/3D_CNN_init_train_data.pt")
torch.save(vectors3D_val,       "Embedding/3D_CNN_init_val_data.pt")
torch.save(vectors3D_test,      "Embedding/3D_CNN_init_test_data.pt")
torch.save(train_labels,        "Embedding/3D_CNN_init_train_labels.pt")
torch.save(val_labels,          "Embedding/3D_CNN_init_val_labels.pt")
torch.save(test_labels,         "Embedding/3D_CNN_init_test_labels.pt")


# Load states of trained models
model_3D = CNN3D_MLP(num_classes=6).to(device)
model_3D.load_state_dict(torch.load("best_model_3D_CNN.pth"))
model_3D.eval()

vectors3D_train = []
vectors3D_val = []
vectors3D_test = []

# Get Train Embeddings from 3D model
with torch.no_grad():
    for inputs, labels in train3D_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        vectors3D_train.append(model_3D.embedding(inputs))
    vectors3D_train = torch.concatenate(vectors3D_train).detach().cpu().numpy()
    print(f'Train Embeddings after 3D Model: {vectors3D_train.shape}')

# Get Validation Embeddings from 3D model
with torch.no_grad():
    for inputs, labels in val3D_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        vectors3D_val.append(model_3D.embedding(inputs))
    vectors3D_val = torch.concatenate(vectors3D_val).detach().cpu().numpy()
    print(f'Val Embeddings after 3D Model: {vectors3D_val.shape}')

# Get Test Embeddings from 3D model
with torch.no_grad():
    for inputs, labels in test3D_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        vectors3D_test.append(model_3D.embedding(inputs))
    vectors3D_test = torch.concatenate(vectors3D_test).detach().cpu().numpy()
    print(f'Test Embeddings after 3D Model: {vectors3D_test.shape}')


# Save trained embeddings from 3D model
torch.save(vectors3D_train,     "Embedding/3D_CNN_pretrained_train_data.pt")
torch.save(vectors3D_val,       "Embedding/3D_CNN_pretrained_val_data.pt")
torch.save(vectors3D_test,      "Embedding/3D_CNN_pretrained_test_data.pt")
torch.save(train_labels,        "Embedding/3D_CNN_pretrained_train_labels.pt")
torch.save(val_labels,          "Embedding/3D_CNN_pretrained_val_labels.pt")
torch.save(test_labels,         "Embedding/3D_CNN_pretrained_test_labels.pt")
