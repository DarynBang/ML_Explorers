import numpy as np
import torch
import matplotlib.pyplot as plt
from glob import glob
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def get_train_data_and_labels(path_3ds):
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

    return train_data, train_labels, val_data, val_labels, test_data, test_labels


def plot_images_with_labels(data, labels, label_map, num_images=10):
    # plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(4, 4, i+1)
        image = data[i]  # Get the i-th image
        label = labels[i]  # Get the i-th label
        plt.imshow(image[:, :, 0], cmap='gray')  # Plot only one channel (28x28x28, take first slice)
        plt.title(f"{list(label_map.keys())[list(label_map.values()).index(label)]}")
        plt.axis('off')
        plt.subplots_adjust(wspace=2, hspace=0)  # Add more horizontal and vertical space
    plt.show()


def plot_slices(num_rows, num_columns, width, height, data):
    """Plot a montage of 20 CT slices"""
    data = np.rot90(np.array(data))
    data = np.transpose(data)
    data = np.reshape(data, (num_rows, num_columns, width, height))
    rows_data, columns_data = data.shape[0], data.shape[1]
    heights = [slc[0].shape[0] for slc in data]
    widths = [slc.shape[1] for slc in data[0]]
    fig_width = 12.0
    fig_height = fig_width * sum(heights) / sum(widths)
    f, axarr = plt.subplots(
        rows_data,
        columns_data,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": heights},
    )
    for i in range(rows_data):
        for j in range(columns_data):
            axarr[i, j].imshow(data[i][j], cmap="gray")
            axarr[i, j].axis("off")
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.show()


# Generate synthetic data with 6 classes with PCA
def generate_synthetic_PCA_data(data, labels, name_embedding = 'flatten'):
  X, y = data,labels
  # Apply PCA (reduce to 2D)
  pca = PCA(n_components=2)
  X_pca = pca.fit_transform(X)

  # Define colors and markers for 6 classes
  colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']
  markers = ['o', 's', 'D', 'v', '^', '<']
  labels = [f"Class {i}" for i in range(6)]  # Generate class labels dynamically

  # Plot each class separately
  plt.figure(figsize=(8, 6))
  for i in range(6):
      plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1],
                  color=colors[i], marker=markers[i],
                  label=labels[i], edgecolors='k', alpha=0.7)

  # Labels and title
  plt.xlabel("Principal Component 1")
  plt.ylabel("Principal Component 2")
  plt.title(f"PCA Visualization of {name_embedding} embedding data")
  plt.legend()
  plt.show()


# Generate synthetic data with 6 classes with TSNE
def generate_synthetic_TSNE_data(data, labels, name_embedding = 'flatten'):
  tsne = TSNE(n_components=2, perplexity=30, random_state=42)
  X_tsne,y = tsne.fit_transform(data),labels

  # Define colors and markers for 6 classes
  colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']
  markers = ['o', 's', 'D', 'v', '^', '<']
  labels = [f"Class {i}" for i in range(6)]  # Generate class labels dynamically

  # Plot each class separately
  plt.figure(figsize=(8, 6))
  for i in range(6):
      plt.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1],
                  color=colors[i], marker=markers[i],
                  label=labels[i], edgecolors='k', alpha=0.7)

  # Labels and title
  plt.xlabel("TSNE Component 1")
  plt.ylabel("TSNE Component 2")
  plt.title(f"TSNE Visualization of {name_embedding} embedding data")
  plt.legend()
  plt.show()


def get_data_from_file(name_embedding = '2D_CNN_init'):
  train_data = torch.load(glob(f'Embedding/*{name_embedding}*train*data*')[0],weights_only=False)
  val_data = torch.load(glob(f'Embedding/*{name_embedding}*val*data*')[0],weights_only=False)
  test_data = torch.load(glob(f'Embedding/*{name_embedding}*test*data*')[0],weights_only=False)
  train_labels = torch.load(glob(f'Embedding/*{name_embedding}*train*labels*')[0],weights_only=False)
  val_labels = torch.load(glob(f'Embedding/*{name_embedding}*val*labels*')[0],weights_only=False)
  test_labels = torch.load(glob(f'Embedding/*{name_embedding}*test*labels*')[0],weights_only=False)
  return train_data, val_data, test_data, train_labels, val_labels, test_labels
