import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, roc_curve, precision_recall_curve)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def RandomForest_gridsearch(embedding_name= 'flatten', load_embeddings=True):
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

  # Random Forest Classifier
  rf = RandomForestClassifier(random_state=42)

  # Hyperparameter grid
  parameters = {
      'n_estimators': [50, 100, 200],
      'max_depth': [3, 5, 10],
      'min_samples_split': [2, 5, 10],
      'min_samples_leaf': [1, 2, 4],
      'bootstrap': [True]
  }

  # Grid Search
  grid_search = GridSearchCV(
      estimator=rf,
      param_grid=parameters,
      scoring='accuracy',
      cv=5,
      verbose=1,
      n_jobs=-1  # Use all processors
  )


  grid_search.fit(X_train, y_train)
  # Predict on test data
  y_pred = grid_search.predict(X_test)


  print(f"\n📌 Best Parameters for '{embedding_name}':")
  for param, value in grid_search.best_params_.items():
      print(f"   {param}: {value}")

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
