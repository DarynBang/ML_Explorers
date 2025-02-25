import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, roc_curve, precision_recall_curve)
from src.utils import get_data_from_file
from src.data.make_dataset import CustomMLPDataset

class Adaptive_MLP(nn.Module):
    def __init__(self, embedding_dim = 2048,num_classes=6):
        super(Adaptive_MLP, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(embedding_dim , 1024),
            nn.ReLU(),
            nn.Dropout(0.5))

        self.fc2 = nn.Sequential(
            nn.Linear(1024 , 512),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc3 = nn.Sequential(
            nn.Linear(512 , 256),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc4 = nn.Sequential(
            nn.Linear(256 , num_classes),
            nn.ReLU(),
            nn.Dropout(0.5))
    def forward(self, embedding):
        # MLP Head|
        x = self.fc1(embedding)
        x = self.fc2(x)
        x = self.fc3(x)  # (batch_size, 6) logits
        x = self.fc4(x)  # (batch_size, 6) logits
        x = F.softmax(x, dim=1)  # Apply softmax across class dimension
        return x
    def evaluation(self, y_pred, y_gt):
       # Accuracy
        accuracy = accuracy_score(y_gt, y_pred)
        # Precision, Recall, and F1-score
        precision = precision_score(y_gt, y_pred, average='weighted')
        recall = recall_score(y_gt, y_pred, average='weighted')
        f1 = f1_score(y_gt, y_pred, average='weighted')
        # Print all metrics
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")


def multilayer_perceptron_algorithm(embedding_name= 'flatten'):
  train_data, val_data, test_data, train_labels, val_labels, test_labels = get_data_from_file(name_embedding = embedding_name)
  train_dataset = CustomMLPDataset(train_data, train_labels)
  val_dataset = CustomMLPDataset(val_data, val_labels)
  test_dataset = CustomMLPDataset(test_data, test_labels)
  train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=512, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)


  # ✅ Model, Loss, Optimizer
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = Adaptive_MLP(embedding_dim = train_data.shape[1], num_classes=6).to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=1e-3)

  # ✅ Training Loop
  num_epochs = 30
  train_losses, val_losses = [], []
  best_val_loss = float("inf")

  for epoch in range(num_epochs):
      model.train()
      running_train_loss = 0.0
      for inputs, labels in train_loader:
          inputs, labels = inputs.to(device), labels.to(device)
          optimizer.zero_grad()
          outputs = model(inputs)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()

          running_train_loss += loss.item()

      avg_train_loss = running_train_loss / len(train_loader)
      train_losses.append(avg_train_loss)

      # ✅ Validation Phase
      model.eval()
      running_val_loss = 0.0
      with torch.no_grad():
          for inputs, labels in val_loader:
              inputs, labels = inputs.to(device), labels.to(device)
              outputs = model(inputs)
              loss = criterion(outputs, labels)
              running_val_loss += loss.item()

      avg_val_loss = running_val_loss / len(val_loader)
      val_losses.append(avg_val_loss)

      print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

      # ✅ Save the best model
      if avg_val_loss < best_val_loss:
          best_val_loss = avg_val_loss
          torch.save(model.state_dict(), "best_model.pth")
          print("✅ Best Model Saved!")

  print("🎉 Training Complete!")

  # ✅ Load Best Model for Testing
  model.load_state_dict(torch.load("best_model.pth"))
  model.eval()

  # ✅ Testing Loop
  test_loss = 0.0
  correct = 0
  total = 0


  y_pred, y_gt = [], []
  with torch.no_grad():
      for inputs, labels in test_loader:
          inputs, labels = inputs.to(device), labels.to(device)
          outputs = model(inputs)
          loss = criterion(outputs, labels)
          test_loss += loss.item()

          # Compute Accuracy
          _, predicted = torch.max(outputs, 1)
          correct += (predicted == labels).sum().item()
          total += labels.size(0)
          y_pred.append(outputs)
          y_gt.append(labels)
  avg_test_loss = test_loss / len(test_loader)
  test_accuracy = 100 * correct / total

  y_pred = torch.cat(y_pred).cpu().numpy()
  y_pred = np.argmax(y_pred, axis=1)
  y_gt = torch.cat(y_gt).cpu().numpy()
  print(y_gt.shape)
  print(y_pred.shape)
  print(f" ============= {embedding_name} ============== ")
  model.evaluation(y_pred,y_gt)

  print(f"🧪 Test Loss: {avg_test_loss:.4f} | Test Accuracy: {test_accuracy:.2f}%")

  # ✅ Plot Training, Validation & Test Loss
  plt.figure(figsize=(8,6))
  plt.plot(range(1, num_epochs+1), train_losses, label="Train Loss", marker="o")
  plt.plot(range(1, num_epochs+1), val_losses, label="Validation Loss", marker="o")
  plt.axhline(avg_test_loss, linestyle="--", color="red", label=f"Test Loss ({avg_test_loss:.4f})")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.title("Training & Validation Loss")
  plt.legend()
  plt.grid()
  plt.show()


if __name__ == '__main__':
  models_name = ['flatten', '2D_CNN_init', '2D_CNN_pretrained', '3D_CNN_init', '3D_CNN_pretrained']
  for model in models_name:
    multilayer_perceptron_algorithm(embedding_name = model)
