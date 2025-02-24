import numpy as np
import torch
from src.data.make_dataset import Custom2DDataset, Custom3DDataset
from src.models.CNN2D import CNN2D_MLP
from src.models.CNN3D import CNN3D_MLP
from torch.nn.utils import DataLoader
from utils import get_train_data_and_labels

path_3ds = glob("MNIST/*3d*")

train_data    = []
val_data      = []
test_data     = []
train_labels  = []
val_labels    = []
test_labels   = []
labels_map    = {}

train_data, train_labels, val_data, val_labels, test_data, test_labels, labels_map = get_train_data_and_labels(path_3ds)


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


def get_embeddings(model, loader):
    vectors = []
        
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            vectors.append(model.get_embedding(inputs))
        vectors = torch.concatenate(vectors).detach().cpu().numpy()

    return vectors
    

def get_embeddings_from_loaders(train_loader, val_loader, test_loader, model, model_path = None, pretrained=False):
    if pretrained and model_path is not None:
        model = model.load_state_dict(torch.load(model_path))
    model.eval()

    train_vectors, val_vectors, test_vectors = get_embeddings(model, train_loader), get_embeddings(model, val_loader), get_embeddings(model, test_loader)
    

    return train_vectors, val_vectors, test_vectors
    

train2D_dataset = Custom2DDataset(train_data, train_labels)
val2D_dataset = Custom2DDataset(val_data, val_labels)
test2D_dataset = Custom2DDataset(test_data, test_labels)

train2D_loader = DataLoader(train2D_dataset, batch_size=512, shuffle=False)
val2D_loader = DataLoader(val2D_dataset, batch_size=512, shuffle=False)
test2D_loader = DataLoader(test2D_dataset, batch_size=512, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
model_2D = CNN2D_MLP(num_classes=6).to(device)
model_2D.eval()

# Save embeddings from the initial 2D data
vectors2D_train, vectors2D_val, vectors2D_test  = get_embeddings_from_loaders(train2D_loader, val2D_loader, test2D_loader, model_2D, None, False)
# Print the embedding shapes
print(f'Train Embeddings after 2D Model: {vectors2D_train.shape}')
print(f'Val Embeddings after 2D Model: {vectors2D_val.shape}')
print(f'Test Embeddings after 2D Model: {vectors2D_test.shape}')

# Save initial 2D data
torch.save(vectors2D_train,     "Embedding/2D_CNN_init_train_data.pt")
torch.save(vectors2D_val,       "Embedding/2D_CNN_init_val_data.pt")
torch.save(vectors2D_test,      "Embedding/2D_CNN_init_test_data.pt")
torch.save(train_labels,        "Embedding/2D_CNN_init_train_labels.pt")
torch.save(val_labels,          "Embedding/2D_CNN_init_val_labels.pt")
torch.save(test_labels,         "Embedding/2D_CNN_init_test_labels.pt")


# Save embeddings from the train 2D model
vectors2D_train, vectors2D_val, vectors2D_test  = get_embeddings_from_loaders(train2D_loader,
                                                                              val2D_loader,
                                                                              test2D_loader,
                                                                              model_2D,
                                                                              "best_model_2D_CNN.pth",
                                                                              True)
# Print the embedding shapes
print(f'Train Embeddings after 2D Model: {vectors2D_train.shape}')
print(f'Val Embeddings after 2D Model: {vectors2D_val.shape}')
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

# Load models
model_3D = CNN3D_MLP(num_classes=6).to(device)
model_3D.eval()

# Save initial 3D embeddings
vectors3D_train, vectors3D_val, vectors3D_test  = get_embeddings_from_loaders(train3D_loader,
                                                                              val3D_loader,
                                                                              test3D_loader,
                                                                              model_3D,
                                                                              None,
                                                                              False)

torch.save(vectors3D_train,     "Embedding/3D_CNN_init_train_data.pt")
torch.save(vectors3D_val,       "Embedding/3D_CNN_init_val_data.pt")
torch.save(vectors3D_test,      "Embedding/3D_CNN_init_test_data.pt")
torch.save(train_labels,        "Embedding/3D_CNN_init_train_labels.pt")
torch.save(val_labels,          "Embedding/3D_CNN_init_val_labels.pt")
torch.save(test_labels,         "Embedding/3D_CNN_init_test_labels.pt")


model_3D = CNN3D_MLP(num_classes=6).to(device)

vectors3D_train, vectors3D_val, vectors3D_test  = get_embeddings_from_loaders(train3D_loader,
                                                                              val3D_loader,
                                                                              test3D_loader,
                                                                              model_3D,
                                                                              "best_model_3D_CNN.pth",
                                                                              True)

# Save trained embeddings from 3D model
torch.save(vectors3D_train,     "Embedding/3D_CNN_pretrained_train_data.pt")
torch.save(vectors3D_val,       "Embedding/3D_CNN_pretrained_val_data.pt")
torch.save(vectors3D_test,      "Embedding/3D_CNN_pretrained_test_data.pt")
torch.save(train_labels,        "Embedding/3D_CNN_pretrained_train_labels.pt")
torch.save(val_labels,          "Embedding/3D_CNN_pretrained_val_labels.pt")
torch.save(test_labels,         "Embedding/3D_CNN_pretrained_test_labels.pt")
