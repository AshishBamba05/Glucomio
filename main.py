import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert the data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)


class DiabetesRiskModel(nn.Module):
    def init(self):
        super(DiabetesRiskModel, self).init()
        self.fc1 = nn.Linear(X_train.shape[1], 64)  # Input layer (number of genotype features)
        self.fc2 = nn.Linear(64, 32)  # Hidden layer
        self.fc3 = nn.Linear(32, 1)   # Output layer (binary classification)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Sigmoid for binary classification
        return x


model = DiabetesRiskModel()
criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)




epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

# Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

#Backward pass and optimization
    loss.backward()
    optimizer.step()

# Print the loss every 10 epochs
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    predictions = (predictions > 0.5).float()  # Convert probabilities to binary output
    accuracy = (predictions == y_test_tensor).sum() / y_test_tensor.shape[0]
    print(f'Accuracy on the test set: {accuracy.item() * 100:.2f}%')

