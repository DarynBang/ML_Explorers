# !pip install hmmlearn 
## Ensure hmmlearn is installed

import numpy as np
import pandas as pd
import pickle
from hmmlearn import hmm
import operator
from copy import copy
from scipy.special import softmax


class HMM_classifier():
    def __init__(self, base_hmm_model):
        self.models = {}
        self.hmm_model = base_hmm_model

    def fit(self, X, Y):
        """
        X: input sequence [[[x1,x2,.., xn]...]]
        Y: output classes [1, 2, 1, ...]
        """

        X_Y = {}
        X_lens = {}
        for c in set(Y):
            X_Y[c] = []
            X_lens[c] = []

        for x, y in zip(X, Y):
            X_Y[y].extend(x)
            X_lens[y].append(len(x))

        for c in set(Y):
            print("Fit HMM for", c, " class")
            hmm_model = copy(self.hmm_model)
            hmm_model.fit(X_Y[c], X_lens[c])
            self.models[c] = hmm_model

    def _predict_scores(self, X):

        """
        X: input sample [[x1,x2,.., xn]]
        Y: dict with log likehood per class
        """
        X_seq = []
        X_lens = []
        for x in X:
            X_seq.extend(x)
            X_lens.append(len(x))

        scores = {}
        for k, v in self.models.items():
            scores[k] = v.score(X)

        return scores

    def predict_proba(self, X):
        """
        X: input sample [[x1,x2,.., xn]]
        Y: dict with probabilities per class
        """
        pred = self._predict_scores(X)

        keys = list(pred.keys())
        scores = softmax(list(pred.values()))

        return dict(zip(keys, scores))

    def predict(self, X):
        """
        X: input sample [[x1,x2,.., xn]]
        Y: predicted class label
        """
        pred = self.predict_proba(X)

        return max(pred.items(), key=operator.itemgetter(1))[0]



def hidden_markov_model_algorithm(embedding_name= 'flatten'):
  train_data, val_data, test_data, train_labels, val_labels, test_labels = get_data_from_file(name_embedding = embedding_name)
  print(f" ============= {embedding_name} ============== ")

  X = np.concatenate((train_data, val_data, test_data), axis=0)
  y = np.concatenate((train_labels, val_labels, test_labels), axis=0)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  clf = HMM_classifier(hmm.CategoricalHMM())
  clf.fit(X_train, y_train)

  print('Got here')
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
