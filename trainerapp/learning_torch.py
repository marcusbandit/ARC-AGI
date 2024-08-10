import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Example: Create dummy data
X = torch.randn(100, 10)  # 100 samples, 10 features each
y = torch.randint(0, 2, (100,))  # 100 labels (binary classification)

# Create a dataset and data loader
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 50)  # 10 input features, 50 hidden units
        self.fc2 = nn.Linear(50, 2)   # 50 hidden units, 2 output classes

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleNN()


criterion = nn.CrossEntropyLoss()  # For classification
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent with learning rate 0.01


num_epochs = 10  # Number of times to go through the dataset

for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        optimizer.zero_grad()  # Zero the parameter gradients

        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute the loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Optimize the weights

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


with torch.no_grad():  # Disable gradient calculation for evaluation
    correct = 0
    total = 0
    for inputs, labels in dataloader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy: {100 * correct / total}%')


torch.save(model.state_dict(), 'simple_nn.pth')