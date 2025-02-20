
import torch
import torch.nn as nn

from CNN2D import CNN2D_MLP
from CNN3D import CNN3D_MLP
from data.make_dataset import Custom2DDataset, Custom3DDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def trainCNN(train_ds, val_ds, test_ds, model, num_epochs=50):
  train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
  val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)
  test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
  
  # ✅ Model, Loss, Optimizer
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = model2D
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=1e-3)
  
  # ✅ Training Loop
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
          torch.save(model.state_dict(), "best_model_2D_CNN.pth")
          print("✅ Best Model Saved!")
  
  print("🎉 Training Complete!")
  
  # ✅ Load Best Model for Testing
  model.load_state_dict(torch.load("best_model_2D_CNN.pth"))
  model.eval()
  
  # ✅ Testing Loop
  test_loss = 0.0
  correct = 0
  total = 0
  
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
  
  avg_test_loss = test_loss / len(test_loader)
  test_accuracy = 100 * correct / total
  
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
  # Train 2D model
  train2D_dataset = Custom2DDataset(train_data, train_labels)
  val2D_dataset = Custom2DDataset(val_data, val_labels)
  test2D_dataset = Custom2DDataset(test_data, test_labels)

  model2D = CNN2D_MLP(num_classes=6).to(device)

  trainCNN(train2D_dataset, val2D_dataset, test2D_dataset, model2D)

  # Train 3D model
  train3D_dataset = Custom3DDataset(train_data, train_labels)
  val3D_dataset = Custom3DDataset(val_data, val_labels)
  test3D_dataset = Custom3DDataset(test_data, test_labels)

  model3D = CNN3D_MLP(num_classes=6).to(device)

  trainCNN(train3D_dataset, val3D_dataset, test3D_dataset, model3D)
  
