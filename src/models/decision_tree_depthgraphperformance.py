import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, roc_curve, precision_recall_curve)
def decision_tree_algorithm(embedding_name='flatten', max_depths=range(5, 30, 5)):
    train_data, val_data, test_data, train_labels, val_labels, test_labels = get_data_from_file(name_embedding=embedding_name)
    X = np.concatenate((train_data, val_data, test_data), axis=0)
    y = np.concatenate((train_labels, val_labels, test_labels), axis=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    accuracies = []
    for depth in max_depths:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        # Precision, Recall, and F1-score
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f" ============= {embedding_name} ============== ")
        # Print all metrics
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

    # Plotting the graph
    plt.figure(figsize=(10, 6))
    plt.plot(max_depths, accuracies, marker='o')
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.title(f'Decision Tree Accuracy vs. Max Depth for {embedding_name}')
    plt.grid(True)
    plt.show()
    
models_name = ['flatten', '2D_CNN_init', '2D_CNN_pretrained', '3D_CNN_init', '3D_CNN_pretrained']
for model in models_name:
  decision_tree_algorithm(embedding_name= model)