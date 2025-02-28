import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,
from src.utils import get_data_from_file
import pandas as pd
from sklearn.svm import SVC

def svm_algorithm(embedding_name= 'flatten', load_embeddings=True):
  train_data, val_data, test_data, train_labels, val_labels, test_labels = get_data_from_file(name_embedding = embedding_name)
  print(f" ============= {embedding_name} ============== ")

  X = np.concatenate((train_data, val_data, test_data), axis=0)
  y = np.concatenate((train_labels, val_labels, test_labels), axis=0)

  if X.shape[1] > 500 and not load_embeddings:
      print("Reducing Dimensionality with TSNE")
      tsne = TSNE(n_components=3, perplexity=30, random_state=42)
      X = tsne.fit_transform(X)

  else:
      save_path = f"/content/drive/MyDrive/TSNE_{embedding_name}_data.npy"
      X = np.load(save_path)

      print(f'Loaded embeddings from path {save_path}!')


  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  clf = SVC(gamma='auto')
  clf.fit(X_train, y_train)
  # Predict on test data
  y_pred = clf.predict(X_test)

  # Accuracy
  accuracy = accuracy_score(y_test, y_pred)
  # Precision, Recall, and F1-score
  precision = precision_score(y_test, y_pred, average='weighted')
  recall = recall_score(y_test, y_pred, average='weighted')
  f1 = f1_score(y_test, y_pred, average='weighted')
  # Print all metrics
  print(f"Accuracy: {accuracy:.4f}")
  print(f"Precision: {precision:.4f}")
  print(f"Recall: {recall:.4f}")
  print(f"F1 Score: {f1:.4f}")

if __name__ == '__main__':
  models_name = ['flatten', '2D_CNN_init', '2D_CNN_pretrained', '3D_CNN_init', '3D_CNN_pretrained']
  for model in models_name:
    svm_algorithm(embedding_name = model)
