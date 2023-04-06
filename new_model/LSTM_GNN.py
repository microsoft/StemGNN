import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class DataLoader:
    def __init__(self, data_path, start_day, n_training, n_timesteps, n_pred):
        self.x = pd.read_csv(data_path).to_numpy()
        self.start_day = start_day
        self.n_timesteps = n_timesteps
        self.n_training = n_training
        self.n_pred = n_pred
        self.n_batch = n_training-n_timesteps-n_pred

    def train_test_split(self):
        self.train = self.x[:,self.start_day:self.start_day+self.n_training]
        self.scale = StandardScaler()
        self.train = self.scale.fit_transform(self.train.T).T
        self.x_train = np.zeros(shape=(self.n_batch, self.n_timesteps, 77))
        self.y_train = np.zeros(shape=(self.n_batch, 77*self.n_pred))
        for i in range(self.n_batch):
            self.x_train[i,:,:] = self.train[:,i:i+self.n_timesteps].T
            self.y_train[i,:] = self.train[:,i+self.n_timesteps:i+self.n_timesteps+self.n_pred].flatten()

        # +1 is added here to resolve the issue that the state of LSTM is 0 for the first element
        self.test = self.x[:,self.start_day+self.n_pred+1:]
        self.test = self.scale.transform(self.test.T).T
        self.x_test = np.zeros(shape=(self.n_batch, self.n_timesteps, 77))
        for i in range(self.n_batch):
            self.x_test[i,:,:] = self.test[:,i:i+self.n_timesteps].T
        return tf.convert_to_tensor(self.x_train), tf.convert_to_tensor(self.y_train), tf.convert_to_tensor(self.x_test)

class LSTMoutput:
    def __init__(self, DataLoader, airport_index):
        self.actual = DataLoader.x
        self.n_timesteps = DataLoader.n_timesteps
        self.scale = DataLoader.scale
        self.airport_index = airport_index

        self.n_batch = DataLoader.n_batch
        self.n_training = DataLoader.n_training
        self.n_pred = DataLoader.n_pred
        self.start_day = DataLoader.start_day

    def inverse_transform(self, input):
        self.predicted_transformed = self.scale.inverse_transform(input[-1,:].reshape(77,self.n_pred).T).T[self.airport_index,:]

        self.predicted_transformed = []
        for i in range(self.n_batch):
            traffic = self.scale.inverse_transform(input[i,:].reshape(77,self.n_pred).T).T[self.airport_index, self.n_pred-1]
            self.predicted_transformed.append(traffic)

    def plot(self):
        x_train_start = self.start_day
        y_train_start = self.start_day
        x_train_end = x_train_start + self.n_training
        y_train_end = y_train_start + self.n_training

        x_test_start = self.start_day+self.n_training-1
        x_test_end = x_test_start+self.n_pred+1
        y_test_start = self.start_day+self.n_training-1
        y_test_end = x_test_start+self.n_pred+1

        x_forecast_start = 2*self.n_pred+self.start_day+self.n_timesteps
        x_forecast_end = x_forecast_start + self.n_batch
        
        fig, ax = plt.subplots()
        sns.lineplot(x=np.arange(x_train_start, x_train_end), 
                     y=self.actual[self.airport_index, np.arange(y_train_start, y_train_end)], 
                     label='Training',
                     ax=ax)
        sns.lineplot(x=np.arange(x_test_start, x_test_end), 
                     y=self.actual[self.airport_index, np.arange(y_test_start, y_test_end)], 
                     label='Testing - Actual',
                     ax=ax)
        sns.lineplot(x=np.arange(x_forecast_start, x_forecast_end), 
                     y=self.predicted_transformed, 
                     color='red', 
                     marker='o', 
                     ms=5, 
                     label='Testing - Predicted',
                     ax=ax)
        ax.set(title='LSTM Forecast for Throughout at ATL',
               ylim=(0,1600), 
               xlim=(x_train_start, x_forecast_end+7))
        plt.axvline(x=x_train_end-1, color='black', linestyle='--', linewidth=0.75)
        plt.legend()
