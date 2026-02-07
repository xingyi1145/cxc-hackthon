import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import os

# Load the dataset
file_path = '../data/bankOfCanadaInterestRateDf.csv'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file {file_path} was not found. Please check the path.")

df = pd.read_csv(file_path)

# Filter for the target interest rate
df = df[df['Financial market statistics'] == 'Overnight money market financing']

# Drop unnecessary columns and handle missing values
df = df[['REF_DATE', 'VALUE']].dropna()

# Convert REF_DATE to datetime and sort
df['REF_DATE'] = pd.to_datetime(df['REF_DATE'])
df = df.sort_values('REF_DATE')

# Use the 'VALUE' column for training
data = df['VALUE'].values.astype(float)

# Normalize the data
scaler = MinMaxScaler(feature_range=(-1, 1))
data_normalized = scaler.fit_transform(data.reshape(-1, 1))

# Split data into training and testing sets (80-20 split)
train_size = int(len(data_normalized) * 0.8)
train_data = data_normalized[:train_size]

# Create sequences for LSTM
def create_sequences(input_data, seq_length):
    sequences = []
    labels = []
    for i in range(len(input_data) - seq_length):
        seq = input_data[i:i+seq_length]
        label = input_data[i+seq_length:i+seq_length+1]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

seq_length = 10
X_train, y_train = create_sequences(train_data, seq_length)

# Convert to PyTorch tensors
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()

# Define the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

# Instantiate the model, define loss function and optimizer
model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 1
for i in range(epochs):
    for seq, labels in zip(X_train, y_train):
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                             torch.zeros(1, 1, model.hidden_layer_size))
        y_pred = model(seq)
        single_loss = loss_function(y_pred, labels.view_as(y_pred))
        single_loss.backward()
        optimizer.step()
    print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

# Save the model
model_path = 'interest_rate_model.onnx'
model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                     torch.zeros(1, 1, model.hidden_layer_size))
dummy_input = torch.randn(seq_length)
torch.onnx.export(model, dummy_input, model_path, input_names=['input'], output_names=['output'])
print(f'Model saved to {model_path}')

torch.save(model.state_dict(), 'interest_rate_model.pth')
print('Model state dict saved to interest_rate_model.pth')

# Save the scaler
import joblib
scaler_path = 'scaler.pkl'
joblib.dump(scaler, scaler_path)
print(f'Scaler saved to {scaler_path}')
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import os

# Define the LSTM model (Must match the architecture in trainInterest.py)
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

def create_sequences(input_data, seq_length):
    sequences = []
    labels = []
    for i in range(len(input_data) - seq_length):
        seq = input_data[i:i+seq_length]
        label = input_data[i+seq_length:i+seq_length+1]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

def main():
    # 1. Load and Preprocess Data
    file_path = '../data/bankOfCanadaInterestRateDf.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} was not found.")

    df = pd.read_csv(file_path)
    
    # Filter and clean data (same steps as training)
    df = df[df['Financial market statistics'] == 'Overnight money market financing']
    df = df[['REF_DATE', 'VALUE']].dropna()
    df['REF_DATE'] = pd.to_datetime(df['REF_DATE'])
    df = df.sort_values('REF_DATE')
    
    data = df['VALUE'].values.astype(float)

    # 2. Load Scaler and Normalize
    if not os.path.exists('scaler.pkl'):
        raise FileNotFoundError("scaler.pkl not found. Please run trainInterest.py first.")
    
    scaler = joblib.load('scaler.pkl')
    data_normalized = scaler.transform(data.reshape(-1, 1))

    # 3. Prepare Test Data (Last 20%)
    train_size = int(len(data_normalized) * 0.8)
    test_data = data_normalized[train_size:]

    seq_length = 10
    X_test, y_test = create_sequences(test_data, seq_length)

    if len(X_test) == 0:
        print("Not enough data in test set to create sequences.")
        return

    # Convert to PyTorch tensors
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()

    # 4. Load Model
    model = LSTM()
    if not os.path.exists('interest_rate_model.pth'):
         raise FileNotFoundError("interest_rate_model.pth not found. Please run trainInterest.py first.")
    
    model.load_state_dict(torch.load('interest_rate_model.pth'))
    model.eval()

    # 5. Make Predictions
    predictions = []
    with torch.no_grad():
        for seq in X_test:
            # Reset hidden state for each sequence
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                 torch.zeros(1, 1, model.hidden_layer_size))
            predictions.append(model(seq).item())

    # 6. Inverse Transform
    predictions = np.array(predictions).reshape(-1, 1)
    y_test_actual = y_test.numpy().reshape(-1, 1)

    predictions_inv = scaler.inverse_transform(predictions)
    y_test_inv = scaler.inverse_transform(y_test_actual)

    # 7. Calculate Metrics
    r2 = r2_score(y_test_inv, predictions_inv)
    mse = mean_squared_error(y_test_inv, predictions_inv)
    mae = mean_absolute_error(y_test_inv, predictions_inv)

    print("-" * 30)
    print(f"Validation Metrics (Last 20% of Data)")
    print("-" * 30)
    print(f"R^2 Score:             {r2:.4f}")
    print(f"Mean Squared Error:    {mse:.4f}")
    print(f"Mean Absolute Error:   {mae:.4f}")
    print("-" * 30)

    # 8. Plot Results
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_inv, label='Actual Interest Rate', color='blue')
    plt.plot(predictions_inv, label='Predicted Interest Rate', color='orange', linestyle='--')
    plt.title('Interest Rate Prediction Validation')
    plt.xlabel('Time Steps')
    plt.ylabel('Interest Rate (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_plot = 'interest_rate_validation.png'
    plt.savefig(output_plot)
    print(f"Validation graph saved to {output_plot}")
    plt.show()

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import os

# Define the LSTM model (Must match the architecture in trainInterest.py)
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

def create_sequences(input_data, seq_length):
    sequences = []
    labels = []
    for i in range(len(input_data) - seq_length):
        seq = input_data[i:i+seq_length]
        label = input_data[i+seq_length:i+seq_length+1]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

def main():
    # 1. Load and Preprocess Data
    file_path = '../data/bankOfCanadaInterestRateDf.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} was not found.")

    df = pd.read_csv(file_path)
    
    # Filter and clean data (same steps as training)
    df = df[df['Financial market statistics'] == 'Overnight money market financing']
    df = df[['REF_DATE', 'VALUE']].dropna()
    df['REF_DATE'] = pd.to_datetime(df['REF_DATE'])
    df = df.sort_values('REF_DATE')
    
    data = df['VALUE'].values.astype(float)

    # 2. Load Scaler and Normalize
    if not os.path.exists('scaler.pkl'):
        raise FileNotFoundError("scaler.pkl not found. Please run trainInterest.py first.")
    
    scaler = joblib.load('scaler.pkl')
    data_normalized = scaler.transform(data.reshape(-1, 1))

    # 3. Prepare Test Data (Last 20%)
    train_size = int(len(data_normalized) * 0.8)
    test_data = data_normalized[train_size:]

    seq_length = 10
    X_test, y_test = create_sequences(test_data, seq_length)

    if len(X_test) == 0:
        print("Not enough data in test set to create sequences.")
        return

    # Convert to PyTorch tensors
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()

    # 4. Load Model
    model = LSTM()
    if not os.path.exists('interest_rate_model.pth'):
         raise FileNotFoundError("interest_rate_model.pth not found. Please run trainInterest.py first.")
    
    model.load_state_dict(torch.load('interest_rate_model.pth'))
    model.eval()

    # 5. Make Predictions
    predictions = []
    with torch.no_grad():
        for seq in X_test:
            # Reset hidden state for each sequence
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                 torch.zeros(1, 1, model.hidden_layer_size))
            predictions.append(model(seq).item())

    # 6. Inverse Transform
    predictions = np.array(predictions).reshape(-1, 1)
    y_test_actual = y_test.numpy().reshape(-1, 1)

    predictions_inv = scaler.inverse_transform(predictions)
    y_test_inv = scaler.inverse_transform(y_test_actual)

    # 7. Calculate Metrics
    r2 = r2_score(y_test_inv, predictions_inv)
    mse = mean_squared_error(y_test_inv, predictions_inv)
    mae = mean_absolute_error(y_test_inv, predictions_inv)

    print("-" * 30)
    print(f"Validation Metrics (Last 20% of Data)")
    print("-" * 30)
    print(f"R^2 Score:             {r2:.4f}")
    print(f"Mean Squared Error:    {mse:.4f}")
    print(f"Mean Absolute Error:   {mae:.4f}")
    print("-" * 30)

    # 8. Plot Results
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_inv, label='Actual Interest Rate', color='blue')
    plt.plot(predictions_inv, label='Predicted Interest Rate', color='orange', linestyle='--')
    plt.title('Interest Rate Prediction Validation')
    plt.xlabel('Time Steps')
    plt.ylabel('Interest Rate (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_plot = 'interest_rate_validation.png'
    plt.savefig(output_plot)
    print(f"Validation graph saved to {output_plot}")
    plt.show()

if __name__ == "__main__":
    main()
