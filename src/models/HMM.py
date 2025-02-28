# !pip install hmmlearn 
## Ensure hmmlearn is installed

import numpy as np
import pandas as pd
import pickle
from hmmlearn import hmm
import operator


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
      # Compute the likelihood of the symbols for each HMM
      likelihoods = [hmm.score(symbols) for hmm in hmms]
      # Select the class with the highest likelihood
      y_pred.append(np.argmax(likelihoods))
      # Compute the normalized likelihoods as the scores
      score = np.exp(likelihoods) / np.sum(np.exp(likelihoods))
      y_score.append(score)

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
