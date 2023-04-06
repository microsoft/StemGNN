import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class DataLoader:
    def __init__(self, data_path, n_batch, n_timesteps, n_pred):
        self.x = pd.read_csv(data_path).to_numpy()
        self.n_batch = n_batch
        self.n_timesteps = n_timesteps
        self.n_pred = n_pred

    def train_test_split(self):
        self.train = self.x[:,:self.n_batch+self.n_timesteps+self.n_pred]
        self.scale = StandardScaler()
        self.train = self.scale.fit_transform(self.train.T).T
        self.x_train = np.zeros(shape=(self.n_batch, self.n_timesteps, 77))
        self.y_train = np.zeros(shape=(self.n_batch, 77*self.n_pred))
        for i in range(self.n_batch):
            self.x_train[i,:,:] = self.train[:,i:i+self.n_timesteps].T
            self.y_train[i,:] = self.train[:,i+self.n_timesteps:i+self.n_timesteps+self.n_pred].flatten()
        
        self.test = self.x[:,self.n_batch+self.n_timesteps+self.n_pred:]
        self.test = self.scale.transform(self.test.T).T
        self.x_test = np.zeros(shape=(self.n_batch, self.n_timesteps, 77))
        self.y_test = np.zeros(shape=(self.n_batch, 77*self.n_pred))
        for i in range(self.n_batch):
            self.x_test[i,:,:] = self.test[:,i:i+self.n_timesteps].T
            self.y_test[i,:] = self.test[:,i:i+self.n_pred].flatten()

        return tf.convert_to_tensor(self.x_train), tf.convert_to_tensor(self.y_train), tf.convert_to_tensor(self.x_test), tf.convert_to_tensor(self.y_test)

class LSTMoutput:
    def __init__(self, DataLoader, n_batch, n_timesteps, airport_index, series_index):
        self.n_timesteps = n_timesteps
        self.scale = DataLoader.scale
        self.n_batch = n_batch
        self.airport_index = airport_index
        self.series_index = series_index

    def inverse_transform(self, input):
        self.pred_transformed = []
        for i in range(self.n_timesteps):
            traffic = self.scale.inverse_transform(input[i,:].reshape(77,3).T).T[self.airport_index,self.series_index]
            self.pred_transformed.append(traffic)

    def plot(self, actual):
        padded = np.pad(np.array(self.pred_transformed), (self.n_batch, 0))
        self.padded = padded
        value = np.append(padded, actual)
        self.value = value
        day = np.tile(np.arange(1, self.n_timesteps+self.n_batch+1), (2))
        type = np.concatenate((np.array(['Predicted']*(self.n_timesteps+self.n_batch)), np.array(['Actual']*(self.n_timesteps+self.n_batch))))
        atl_plot = pd.DataFrame(np.vstack((value, day, type)).T)
        atl_plot.columns = ['Value', 'Day', 'Type']
        atl_plot[['Value', 'Day']] = atl_plot[['Value', 'Day']].astype(float)
        fig, ax = plt.subplots()
        sns.lineplot(data=atl_plot, x='Day', y='Value', hue='Type', ax=ax)
        ax.set(title='ATL Throughput - Training Results', xlabel='Days Since 1/1/20', ylabel='# of Flights')