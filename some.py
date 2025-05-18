import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

# Load dataset
df = pd.read_csv(r"C:\Users\Dell\Downloads\extended_ecommerce_sales_forecast_with_new_features.csv")

# Encode categorical variables
categorical_columns = ["Seasonal Fluctuations", "Customer Demographics", "Product Categories"]
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Scale numerical features
scaler = MinMaxScaler()
numerical_columns = df.columns.difference(categorical_columns + ["Sales Forecast"])
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Define target and features
target_col = "Sales Forecast"
feature_cols = df.columns.difference([target_col])

# Convert data into sequences
def create_sequences(data, target, seq_length=10):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(target[i+seq_length])
    return np.array(X), np.array(y)

X_data = df[feature_cols].values
y_data = df[target_col].values
X_seq, y_seq = create_sequences(X_data, y_data, seq_length=10)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42, shuffle=False)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out[:, -1, :])
        return self.fc(lstm_out)

# Model parameters
input_size = X_train.shape[2]
hidden_size = 64
num_layers = 2
output_size = 1

# Initialize model, loss function, and optimizer
model = LSTMModel(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 20
batch_size = 32

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# Save model
torch.save(model.state_dict(), "lstm_sales_forecast.pth")

# Evaluate model
model.eval()
y_pred = model(X_test).detach().numpy()
mse = np.mean((y_test.numpy().flatten() - y_pred.flatten())**2)
print(f"Mean Squared Error: {mse}")