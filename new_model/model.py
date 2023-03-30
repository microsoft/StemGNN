import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch
import networkx as nx
from ce290_data.adjacancy import adjacancy
from ce290_data.data_cleaning import data_processing
from scipy.sparse import csgraph


adj = adjacancy('ce290_data/data/apr2020.csv', True)
x_matrix = pd.read_csv('ce290_data/x_matrix/apr2020.csv')

x_matrix_jan = pd.read_csv('ce290_data/x_matrix/jan2020.csv').to_numpy()
x_matrix_feb = pd.read_csv('ce290_data/x_matrix/feb2020.csv').to_numpy()
x_matrix_mar = pd.read_csv('ce290_data/x_matrix/mar2020.csv').to_numpy()
x_matrix_apr = pd.read_csv('ce290_data/x_matrix/apr2020.csv').to_numpy()

x_matrix = np.concatenate((x_matrix_jan, x_matrix_feb, x_matrix_mar, x_matrix_apr), axis=1)
x_matrix = np.nan_to_num(x_matrix, nan=0)


# define the sequence length and output dimension
seq_length = 50
output_dim = 77
T = x_matrix.shape[1]

def create_sequences(data, seq_length, num_out_timesteps):
    X = []
    y = []
    for i in range(len(data) - seq_length - num_out_timesteps + 1):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length:i+seq_length+num_out_timesteps])
    return np.array(X), np.array(y)

# define the number of output time steps to predict
num_out_timesteps = 1

# create training and validation sequences
X_train, y_train = create_sequences(x, seq_length, num_out_timesteps)
X_train = X_train.reshape(71,50,-1)
y_train = y_train.reshape(71,1)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# define your training dataset (assuming you have X_train and y_train data)
train_dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float())

# define the batch size for training
batch_size = 64

# create the train_loader object
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define hyperparameters
input_size = 1
hidden_size = 16
num_layers = 1
output_size = 1
batch_size = 64
num_epochs = 50
learning_rate = 0.001

# Define the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        out, (hn, cn) = self.lstm(x.unsqueeze(2), (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out


# Initialize the model and optimizer
model = LSTM(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    outputs = []
    for i in range(0, train_data.size(0) - batch_size, batch_size):
        input_seq = train_data[i:i+batch_size]
        output_seq = train_data[i+1:i+1+batch_size]
        optimizer.zero_grad()
        model.hidden = (torch.zeros(num_layers, batch_size, hidden_size).requires_grad_(), 
                        torch.zeros(num_layers, batch_size, hidden_size).requires_grad_())
        y_pred = model(input_seq)
        loss = criterion(y_pred, output_seq)
        loss.backward()
        optimizer.step()
        outputs.append(y_pred.detach().numpy())
    print("Epoch:", epoch, " Loss:", loss.item())
