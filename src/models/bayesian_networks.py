# Ensure pgmpy is installed

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,
from src.utils import get_data_from_file
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.inference import VariableElimination


def bayesian_network_algorithm(embedding_name='flatten', load_embeddings=False):
    train_data, val_data, test_data, train_labels, val_labels, test_labels = get_data_from_file(name_embedding=embedding_name)
    print(f" ============= {embedding_name} (Bayesian Network) ============== ")

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

    # Discretize data
    n_bins = 10  # Adjust as needed
    X_discrete = pd.DataFrame()

    for col in range(X.shape[1]):  # Iterate through columns (features)
        X_discrete[f"feature_{col}"] = pd.qcut(X[:, col], q=n_bins, duplicates='drop', labels=False)
    X_discrete['label'] = y  # Add labels

    # Split back into train, val, test
    train_data_discrete = X_discrete.iloc[:len(train_data)]
    val_data_discrete = X_discrete.iloc[len(train_data):len(train_data) + len(val_data)]
    test_data_discrete = X_discrete.iloc[len(train_data) + len(val_data):]

    # Define the Bayesian Network structure (DAG)
    structure = []
    for feature_col in train_data_discrete.columns[:-1]: # exclude labels
        structure.append((feature_col, 'label'))

    model = BayesianNetwork(structure)


    model.fit(train_data_discrete, estimator=MaximumLikelihoodEstimator)

    # Make predictions
    inference = VariableElimination(model)

    predictions = []
    for _, row in test_data_discrete.iterrows():
        evidence = row.to_dict()
        true_label = evidence.pop('label')
        result = inference.map_query(variables=['label'], evidence=evidence)
        # predicted_label = result['label'].values.argmax() #.values to get the array and argmax to find the index of max value
        predictions.append(result['label'])

    # Convert predictions and true labels to the same type
    true_labels = test_data_discrete['label'].astype(str)  # Ensure consistent type
    predictions = np.array(predictions).astype(str)

    # Compute metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)

    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
