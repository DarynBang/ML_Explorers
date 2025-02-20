import glob
import numpy as np
import torch
from utils import plot_images_with_labels, plot_slices

path_3ds = glob("MNIST/*3d*")

train_data    = []
val_data      = []
test_data     = []
train_labels  = []
val_labels    = []
test_labels   = []
labels_map    = {}

for idx, path in enumerate(path_3ds):
  data = np.load(path)
  data_train_images = data["train_images"]
  data_val_images   = data['val_images']
  data_test_images = data["test_images"]
  train_data.append(data_train_images)
  train_labels.append(np.ones(len(data_train_images))*idx)
  val_data.append(data_val_images)
  val_labels.append(np.ones(len(data_val_images))*idx)
  test_data.append(data_test_images)
  test_labels.append(np.ones(len(data_test_images))*idx)
  labels_map[path.split("/")[-1].split(".")[0]] = idx

train_data = np.concatenate(train_data)
train_labels = np.concatenate(train_labels)
val_data = np.concatenate(val_data)
val_labels = np.concatenate(val_labels)
test_data = np.concatenate(test_data)
test_labels = np.concatenate(test_labels)
# Shuffle the data and labels
train_data, train_labels = shuffle(train_data, train_labels, random_state=42)
val_data, val_labels = shuffle(val_data, val_labels, random_state=42)
test_data, test_labels = shuffle(test_data, test_labels, random_state=42)

# Plot a few images 
plot_images_with_labels(train_data, train_labels, labels_map, num_images=16)

def plot_medMnist3D_images(file_path):
  # Plot images from file path
  data = np.load(file_path)
  train_images = data["train_images"]
  images = train_images
  image  = images[0]
  print("Dimension of the CT scan is:", image.shape)
  plt.imshow(np.squeeze(image[:, :, 20]), cmap="gray")
  plot_slices(4, 7, 28, 28, image[:, :, :40])

if __name__ == '__main__':
  medMnist3d_file_paths = ['MNIST/synapsemnist3d.npz', 'MNIST/nodulemnist3d.npz', 'MNIST/adrenalmnist3d.npz', 'MNIST/fracturemnist3d.npz', 'MNIST/organmnist3d.npz']

  for file_path in medMnist3d_file_paths:
    plot_medMnist3D_images(file_path)
