import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

class DataLoader:
    def __init__(self, data_path, start_day, n_training, n_timesteps, n_pred, day_start, day_end):
        self.x = pd.read_csv(data_path).to_numpy()
        self.start_day = start_day
        self.n_timesteps = n_timesteps
        self.n_training = n_training
        self.n_pred = n_pred
        self.n_batch = n_training-n_timesteps-n_pred
        self.day_start = day_start
        self.day_end = day_end
        self.airport_list = ['ABQ', 'ANC', 'ATL', 'AUS', 'BDL', 'BHM', 'BNA', 'BOS', 'BUF', 'BUR', 'BWI',
                     'CLE', 'CLT', 'CVG', 'DAL', 'DAY', 'DCA', 'DEN', 'DFW', 'DTW', 'EWR', 'FLL', 'GYY',
                     'HNL', 'HOU', 'HPN', 'IAD', 'IAH', 'IND', 'ISP', 'JAX', 'JFK', 'LAS', 'LAX', 'LGA', 'LGB',
                     'MCI', 'MCO', 'MDW', 'MEM', 'MHT', 'MIA', 'MKE', 'MSP', 'MSY', 'OAK', 'OGG', 'OMA', 'ONT', 'ORD', 'OXR',
                     'PBI', 'PDX', 'PHL', 'PHX', 'PIT', 'PSP', 'PVD', 'RDU', 'RFD', 'RSW',
                     'SAN', 'SAT', 'SDF', 'SEA', 'SFO', 'SJC', 'SJU', 'SLC', 'SMF', 'SNA', 'STL', 'SWF',
                     'TEB', 'TPA', 'TUS', 'VNY']

    def creat_adjacency_matrix(self, df, timesteps_i):
        # Create adjacency matrix grouped by every 28 days
        df1 = df[(df['Day_of_Year'] <= self.day_end) & (df['Day_of_Year'] >= self.day_start)]
        data4adj = df1[(df1['Day_of_Year'] <= self.start_day+timesteps_i+28) & (df1['Day_of_Year'] >= self.start_day+timesteps_i)]
        num_airports = len(self.airport_list)
        adjcc_matrix = np.zeros((num_airports, num_airports))
        
        for _, row in data4adj.iterrows():
            departure_index = self.airport_list.index(row['Departure'])
            arrival_index = self.airport_list.index(row['Arrival'])
            adjcc_matrix[departure_index, arrival_index] += 1

        scaler_adj = MinMaxScaler(feature_range=(0, 1))
        # Reshape the matrix to be treated as a whole
        matrix_reshaped = adjcc_matrix.reshape(-1, 1)
        normalized_matrix_reshaped = scaler_adj.fit_transform(matrix_reshaped)
        # Reshape the normalized matrix back to its original shape
        normalized_matrix = normalized_matrix_reshaped.reshape(adjcc_matrix.shape)
        return normalized_matrix

    def train_test_split(self, combined_data=None):
        self.train = self.x[:,self.start_day:self.start_day+self.n_training]
        self.scale = StandardScaler()
        self.train = self.scale.fit_transform(self.train.T).T
        self.x_train = np.zeros(shape=(self.n_batch, self.n_timesteps, 77))
        self.y_train = np.zeros(shape=(self.n_batch, 77*self.n_pred))
        for i in range(self.n_batch):
            self.x_train[i,:,:] = self.train[:,i:i+self.n_timesteps].T
            self.y_train[i,:] = self.train[:,i+self.n_timesteps:i+self.n_timesteps+self.n_pred].flatten()
            # Multiply the adjacency matrix with the number of flights
            if combined_data is not None:
                xy_adj= self.creat_adjacency_matrix(df=combined_data, timesteps_i=i)
                self.x_train[i,:,:] = self.x_train[i,:,:] @ xy_adj
        # +1 is added here to resolve the issue that the state of LSTM is 0 for the first element
        self.test = self.x[:,self.start_day+self.n_pred+1:]
        self.test = self.scale.transform(self.test.T).T
        self.x_test = np.zeros(shape=(self.n_batch, self.n_timesteps, 77))
        for i in range(self.n_batch):
            self.x_test[i,:,:] = self.test[:,i:i+self.n_timesteps].T
            if combined_data is not None:
                self.x_test[i,:,:] = self.x_test[i,:,:] @ xy_adj
        if combined_data is None:
            return tf.convert_to_tensor(self.x_train), tf.convert_to_tensor(self.y_train), tf.convert_to_tensor(self.x_test)
        else:
            return tf.convert_to_tensor(self.x_train), tf.convert_to_tensor(self.y_train), tf.convert_to_tensor(self.x_test), xy_adj

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

    def get_errors(self):
        return np.array(self.predicted_transformed)[-self.n_pred-1:] - self.actual[self.airport_index, np.arange(self.start_day+self.n_training-1, self.start_day+self.n_training-1+self.n_pred+1)]

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
        ax.set(title=f"""LSTM Forecast for Throughput at {self.airport_list[self.airport_index]}""",
               ylim=(0,1600), 
               xlim=(x_train_start, x_forecast_end+7))
        plt.axvline(x=x_train_end-1, color='black', linestyle='--', linewidth=0.75)
        plt.legend()

def aggregate_errors(testing_time_periods, n_replications, n_training, n_timestep, n_pred, adjacancy_matrix, airport_list1=None, airport_list2=None, gnn=True):
    start_day = testing_time_periods - n_training - n_pred
    rmse = np.empty(shape=(n_replications, start_day.shape[0]))
    mae = np.empty(shape=(n_replications, start_day.shape[0]))
    rmse_l1 = np.empty(shape=(n_replications, start_day.shape[0]))
    mae_l1 = np.empty(shape=(n_replications, start_day.shape[0]))
    rmse_l2 = np.empty(shape=(n_replications, start_day.shape[0]))
    mae_l2= np.empty(shape=(n_replications, start_day.shape[0]))
    for i in range(start_day.shape[0]):
        loader = DataLoader('2020.csv', start_day[i], n_training, n_timestep, n_pred, 1, 366)
        if gnn:
            x_train, y_train, x_test, adj_matrix = loader.train_test_split(combined_data=adjacancy_matrix)
        else:
            x_train, y_train, x_test = loader.train_test_split()
        for j in range(n_replications):
            model = Sequential()
            model.add(LSTM(loader.n_pred*77, batch_input_shape=(loader.n_batch,loader.n_timesteps,77)))
            model.add(Dense(loader.n_pred*77//2, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(loader.n_pred*77, activation='linear'))
            model.compile(loss='mean_squared_error', optimizer='adam')
            model.fit(x_train, y_train, epochs=100, batch_size=14)
            predicted = model.predict(x_test, batch_size=14)
            error = np.empty(1)
            error_airportl1 = np.empty(1)
            error_airportl2 = np.empty(1)
            for k in range(77):
                result = LSTMoutput(loader, k)
                result.inverse_transform(predicted)
                error = np.append(error, [result.get_errors()])

            for k2 in airport_list1:
                result = LSTMoutput(loader, k2)
                result.inverse_transform(predicted)
                error_airportl1 = np.append(error, [result.get_errors()])

            for k3 in airport_list2:
                result = LSTMoutput(loader, k3)
                result.inverse_transform(predicted)
                error_airportl2 = np.append(error, [result.get_errors()])

            error = error[1:]
            error_airportl1 = error_airportl1[1:]
            error_airportl2 = error_airportl2[1:]

            rmse[j,i] = np.sum(error**2) # j replication, testing time frame i
            mae[j,i] = np.sum(np.abs(error))
            rmse_l1[j,i] = np.sum(error_airportl1**2) # j replication, testing time frame i
            mae_l1[j,i] = np.sum(np.abs(error_airportl1))
            rmse_l2[j,i] = np.sum(error_airportl2**2) # j replication, testing time frame i
            mae_l2[j,i] = np.sum(np.abs(error_airportl2))

    rmse = np.sqrt(rmse.mean(axis=1)/(77*n_pred*len(testing_time_periods)))
    mae = mae.mean(axis=1)/(77*n_pred*len(testing_time_periods))
    rmse_l1 = np.sqrt(rmse_l1.mean(axis=1)/(77*n_pred*len(testing_time_periods)))
    mae_l1 = mae_l1.mean(axis=1)/(77*n_pred*len(testing_time_periods))
    rmse_l2 = np.sqrt(rmse_l2.mean(axis=1)/(77*n_pred*len(testing_time_periods)))
    mae_l2 = mae_l2.mean(axis=1)/(77*n_pred*len(testing_time_periods))

    return rmse, mae, rmse_l1, mae_l1, rmse_l2, mae_l2

    