# !pip install hmmlearn 
## Ensure hmmlearn is installed

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,
from src.utils import get_data_from_file
import pandas as pd
import pickle
from hmmlearn import hmm


def hidden_markov_model_algorithm(embedding_name= 'flatten'):
  train_data, val_data, test_data, train_labels, val_labels, test_labels = get_data_from_file(name_embedding = embedding_name)
  print(f" ============= {embedding_name} ============== ")

  X = np.concatenate((train_data, val_data, test_data), axis=0)
  y = np.concatenate((train_labels, val_labels, test_labels), axis=0)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
  hmms = []
  for c in range(num_states):
      # Get the training images for the class
      class_images = X_train[y_train == c]
      # Create a Gaussian HMM with num_states states and diagonal covariance matrix
      model = hmm.GaussianHMM(n_components=num_states, n_iter=1000)
      # Fit the model to the training images
      model.fit(class_images)
      hmms.append(model)

  # Classify the test images using the trained HMMs
  y_pred = []
  y_score = []
  for symbols in X_test:
      likelihoods = [hmm.score(symbols.reshape(1, -1)) for hmm in hmms]
      # Select the class with the highest likelihood
      y_pred.append(np.argmax(likelihoods))

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
    hidden_markov_model_algorithm(embedding_name = model)
