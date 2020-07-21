from utils.math_utils import z_score

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# import numpy as np


class Dataset(object):
    def __init__(self, data, stats):
        self.__data = data
        self.stats = stats
        # self.mean = stats['mean']
        # self.std = stats['std']

    def get_data(self, type):
        return self.__data[type]

    def get_stats(self):
        return self.stats  # {'mean': self.mean, 'std': self.std}

    def get_len(self, type):
        return len(self.__data[type])

    # def z_inverse(self, type):
    #     return self.__data[type] * self.std + self.mean


def seq_gen(len_seq, data_seq, offset, n_frame, n_route, C_0=1):
    '''
    Generate data in the form of standard sequence unit.
    :param len_seq: int, the length of target date sequence.
    :param data_seq: np.ndarray, source data / time-series.
    :param offset:  int, the starting index of different dataset type.
    :param n_frame: int, the number of frame within a standard sequence unit,
                         which contains n_his = 12 and n_pred = 9 (3 /15 min, 6 /30 min & 9 /45 min).
    :param n_route: int, the number of routes in the graph.
    :param day_slot: int, the number of time slots per day, controlled by the time window (5 min as default).
    :param C_0: int, the size of input channel.
    :return: np.ndarray, [len_seq, n_frame, n_route, C_0].
    '''
    samples = np.asarray([np.asarray(data_seq[i:i + n_frame]).reshape([n_frame, n_route, C_0])
                          for i in range(offset, offset + len_seq)])

    return samples
    # n_slot = day_slot - n_frame + 1
    # if len_seq < 1:
    #     if len(data_seq) / n_slot < 50:
    #         total = int(len(data_seq) / n_slot) - 5
    #     else:
    #         total = int(len(data_seq) / n_slot) - 6
    #     len_seq = int(total * len_seq)
    #     offset = int(total * offset)
    #
    # tmp_seq = np.zeros((len_seq * n_slot, n_frame, n_route, C_0))
    # for i in range(len_seq):
    #     for j in range(n_slot):
    #         sta = (i + offset) * day_slot + j
    #         end = sta + n_frame
    #         tmp_seq[i * n_slot + j, :, :, :] = np.reshape(data_seq[sta:end, :], [n_frame, n_route, C_0])


# def seq_gen(len_seq, data_seq, offset, n_frame, n_route, day_slot, C_0=1):
#     '''
#     Generate data in the form of standard sequence unit.
#     :param len_seq: int, the length of target date sequence.
#     :param data_seq: np.ndarray, source data / time-series.
#     :param offset:  int, the starting index of different dataset type.
#     :param n_frame: int, the number of frame within a standard sequence unit,
#                          which contains n_his = 12 and n_pred = 9 (3 /15 min, 6 /30 min & 9 /45 min).
#     :param n_route: int, the number of routes in the graph.
#     :param day_slot: int, the number of time slots per day, controlled by the time window (5 min as default).
#     :param C_0: int, the size of input channel.
#     :return: np.ndarray, [len_seq, n_frame, n_route, C_0].
#     '''
#     n_slot = day_slot - n_frame + 1
#     if len_seq < 1:
#         if len(data_seq) / n_slot < 50:
#             total = int(len(data_seq) / n_slot) - 5
#         else:
#             total = int(len(data_seq) / n_slot) - 6
#         len_seq = int(total * len_seq)
#         offset = int(total * offset)
#
#     tmp_seq = np.zeros((len_seq * n_slot, n_frame, n_route, C_0))
#     for i in range(len_seq):
#         for j in range(n_slot):
#             sta = (i + offset) * day_slot + j
#             end = sta + n_frame
#             tmp_seq[i * n_slot + j, :, :, :] = np.reshape(data_seq[sta:end, :], [n_frame, n_route, C_0])
#     return tmp_seq


def data_gen(file_path, n_route, train_val_test_ratio, scalar, n_frame):
    '''
    Source file load and dataset generation.
    :param file_path: str, the file path of data source.
    :param data_config: tuple, the configs of dataset in train, validation, test.
    :param n_route: int, the number of routes in the graph.
    :param n_frame: int, the number of frame within a standard sequence unit,
                         which contains n_his = 12 and n_pred = 9 (3 /15 min, 6 /30 min & 9 /45 min).
    :param day_slot: int, the number of time slots per day, controlled by the time window (5 min as default).
    :return: dict, dataset that contains training, validation and test with stats.
    '''
    n_train, n_val, n_test = train_val_test_ratio
    # generate training, validation and test data
    data_seq = None
    raw_train_data, raw_inference_data = None, None
    if str.endswith(file_path, 'csv'):
        data_seq = pd.read_csv(file_path, header=None)
        print(data_seq.shape)
        length = len(data_seq) - n_frame + 1
        train_len = int(n_train * length)
        val_len = int(n_val * length)
        test_len = int(n_test * length)
        seq_train = data_seq[:train_len]
    else:
        raw_train_data = pd.read_csv(f'{file_path}/train.csv', header=None)
        raw_inference_data = pd.read_csv(f'{file_path}/inference.csv', header=None)
        train_len = (int)((len(raw_train_data) - n_frame + 1) * 0.75)
        val_len = (int)((len(raw_train_data) - n_frame + 1) * 0.25)
        test_len = len(raw_inference_data) - n_frame + 1
        seq_train = raw_train_data[:train_len]

    if scalar == 'min_max':
        my_matrix = np.array(seq_train)
        scaler = MinMaxScaler()
        scaler.fit(my_matrix)
        seq_train = scaler.transform(my_matrix)

    x_stats = []
    data_seq2 = pd.DataFrame(seq_train)

    if scalar == 'z_score':
        for column in list(data_seq2.columns):
            data = np.array(data_seq2[column])
            stats = {'mean': np.mean(data), 'std': np.std(data)}
            x_stats.append(stats)
    else:
        for column in list(data_seq2.columns):
            stats = {'mean': 0, 'std': 1}
            x_stats.append(stats)

    if data_seq is not None:
        normalized_whole = z_score(data_seq.values, x_stats)
        seq_train = seq_gen(train_len, normalized_whole, 0, n_frame, n_route)
        seq_val = seq_gen(val_len, normalized_whole, train_len, n_frame, n_route)
        seq_test = seq_gen(test_len, normalized_whole, train_len + val_len, n_frame, n_route)
        raw_val = seq_gen(val_len, data_seq, train_len, n_frame, n_route)
        raw_test = seq_gen(test_len, data_seq, train_len + val_len, n_frame, n_route)
    else:
        normalized_train = z_score(raw_train_data.values, x_stats)
        normalized_inference = z_score(raw_inference_data.values, x_stats)
        seq_train = seq_gen(train_len, normalized_train, 0, n_frame, n_route)
        seq_val = seq_gen(val_len, normalized_train, train_len, n_frame, n_route)
        seq_test = seq_gen(test_len, normalized_inference, 0, n_frame, n_route)
        raw_val = seq_gen(val_len, raw_train_data, train_len, n_frame, n_route)
        raw_test = seq_gen(test_len, raw_inference_data, 0, n_frame, n_route)

    x_data = {'train': seq_train, 'val': seq_val, 'test': seq_test, 'val_raw': raw_val, 'test_raw': raw_test}
    dataset = Dataset(x_data, x_stats)
    return dataset


def gen_batch(inputs, batch_size, dynamic_batch=False, shuffle=False):
    '''
    Data iterator in batch.
    :param inputs: np.ndarray, [len_seq, n_frame, n_route, C_0], standard sequence units.
    :param batch_size: int, the size of batch.
    :param dynamic_batch: bool, whether changes the batch size in the last batch if its length is less than the default.
    :param shuffle: bool, whether shuffle the batches.
    '''
    len_inputs = len(inputs)

    if shuffle:
        idx = np.arange(len_inputs)
        np.random.shuffle(idx)

    for start_idx in range(0, len_inputs, batch_size):
        end_idx = start_idx + batch_size
        if end_idx > len_inputs:
            if dynamic_batch:
                end_idx = len_inputs
            else:
                break
        if shuffle:
            slide = idx[start_idx:end_idx]
        else:
            slide = slice(start_idx, end_idx)

        yield inputs[slide]
