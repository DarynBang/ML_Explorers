import numpy as np
import torch
import matplotlib.pyplot as plt
from glob import glob
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


# Generate synthetic data with 6 classes
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

# Generate synthetic data with 6 classes
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
