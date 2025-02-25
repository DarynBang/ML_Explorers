import glob
import numpy as np
import torch
from src.utils import get_train_data_and_labels, plot_images_with_labels, plot_slices

path_3ds = glob("MNIST/*3d*")

train_data    = []
val_data      = []
test_data     = []
train_labels  = []
val_labels    = []
test_labels   = []
labels_map    = {}

train_data, train_labels, val_data, val_labels, test_data, test_labels, labels_map = get_train_data_and_labels(path_3ds)

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
