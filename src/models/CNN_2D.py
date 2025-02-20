
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN2D_MLP(nn.Module):
    def __init__(self, num_classes=6):
        super(CNN2D_MLP, self).__init__()

        # CNN Backbone
        self.conv1 = nn.Conv2d(in_channels=28, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # MLP Head (Flatten and Fully Connected Layers)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)  # Output 6 logits

        self.fc1 = nn.Sequential(
            nn.Linear(1024 , 512),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc2 = nn.Sequential(
            nn.Linear(512 , 256),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc3 = nn.Sequential(
            nn.Linear(256 , num_classes),
            nn.ReLU(),
            nn.Dropout(0.5))


    def forward(self, x):
        # CNN Feature Extraction
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))

        # Flatten (batch, 1024, 1, 1) → (batch, 1024)
        x = torch.flatten(x, start_dim=1)

        # MLP Head
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)  # (batch_size, 6) logits
        x = F.softmax(x, dim=1)  # Apply softmax across class dimension

        return x
    def get_embedding(self, x):
      # CNN Feature Extraction
      x = F.relu(self.conv1(x))
      x = self.pool(F.relu(self.conv2(x)))
      x = self.pool(F.relu(self.conv3(x)))
      x = self.pool(F.relu(self.conv4(x)))
      x = self.pool(F.relu(self.conv5(x)))

      # Flatten (batch, 1024, 1, 1) → (batch, 1024)
      x = torch.flatten(x, start_dim=1)
      return x
  
# Example usage
if __name__ == '__main__':
  batch_size = 8
  x = torch.randn(batch_size, 28, 28, 28)  # (batch_size, 28, 28, 28)
  x = x.permute(0, 3, 1, 2)  # Change to (batch_size, channels, height, width)
  
  model = CNN2D_MLP(num_classes=6)
  output = model(x)
  embedding = model.get_embedding(x)
  print(output.shape)  # Expected output: (batch_size, 6)
  print(embedding.shape)  # Expected output: (batch_size, 1024)
