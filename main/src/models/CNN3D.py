import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN3D_MLP(nn.Module):
    def __init__(self, num_classes=6):
        super(CNN3D_MLP, self).__init__()

        # 3D CNN Backbone
        self.group1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 2, 2)))

        self.group2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 2, 2)))

        self.group3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 2, 2)))

        self.group4 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=2),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 2, 2)))

        self.group5 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, padding=2),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2)))

        # MLP Head (Flatten and Fully Connected Layers)
        self.fc1 = nn.Linear(11776, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, num_classes)  # Output 6 logits

    def forward(self, x):
        # 3D CNN Feature Extraction
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.group4(x)
        x = self.group5(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x
      
    def embedding(self,x):
        # CNN Feature Extraction
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.group4(x)
        x = self.group5(x)
        x = torch.flatten(x, start_dim=1)
        return x


if __name__ == '__main__':
  # Example usage
  batch_size = 8
  x = torch.randn(8, 28, 28, 28)  # (batch_size, 28, 28, 28)
  x = x.unsqueeze(1)  # Add a channel dimension → (batch_size, 28, 28, 28, 28)
  
  model = CNN3D_MLP(num_classes=6)
  output = model(x)
  
  
  print(output.shape)  # Expected output: (batch_size, 6)
  print(embedding.shape)
