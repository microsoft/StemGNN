from data_loader.data_utils import gen_batch
from utils.math_utils import evaluation,z_inverse
from os.path import join as pjoin

import tensorflow as tf
import numpy as np
import time
import pandas as pd

tf.disable_eager_execution()


def z_score(x, mean, std):
    '''
    Z-score normalization function: $z = (X - \mu) / \sigma $,
    where z is the z-score, X is the value of the element,
    $\mu$ is the population mean, and $\sigma$ is the standard deviation.
    :param x: np.ndarray, input array to be normalized.
    :param mean: float, the value of mean.
    :param std: float, the value of standard deviation.
    :return: np.ndarray, z-score normalized array.
    '''
    # return x
    return (x - mean) / std


def z_inverse(x, mean, std):
    '''
    The inverse of function z_score().
    :param x: np.ndarray, input to be recovered.
    :param mean: float, the value of mean.
    :param std: float, the value of standard deviation.
    :return: np.ndarray, z-score inverse array.
    '''

    # return x

    return x * std + mean


def multi_pred(sess, y_pred, seq, batch_size, n_his, n_pred, step_idx, dynamic_batch=True):
    '''
    Multi_prediction function.
    :param sess: tf.Session().
    :param y_pred: placeholder.
    :param seq: np.ndarray, [len_seq, n_frame, n_route, C_0].
    :param batch_size: int, the size of batch.
    :param n_his: int, size of historical records for training.
    :param n_pred: int, the length of prediction.
    :param step_idx: int or list, index for prediction slice.
    :param dynamic_batch: bool, whether changes the batch size in the last one if its length is less than the default.
    :return y_ : tensor, 'sep' [len_inputs, n_route, 1]; 'merge' [step_idx, len_inputs, n_route, 1].
            len_ : int, the length of prediction.
    '''
    pred_list = []
    for i in gen_batch(seq, min(batch_size, len(seq)), dynamic_batch=dynamic_batch):
        # Note: use np.copy() to avoid the modification of source data.
        test_seq = np.copy(i[:, 0:n_his + 1, :, :])
        step_list = []
        for j in range(n_pred):
            pred = sess.run(y_pred,
                            feed_dict={'data_input:0': test_seq, 'keep_prob:0': 1.0})
            if isinstance(pred, list):
                pred = np.array(pred[0])
            test_seq[:, 0:n_his - 1, :, :] = test_seq[:, 1:n_his, :, :]
            test_seq[:, n_his - 1, :, :] = pred
            step_list.append(pred)
        pred_list.append(step_list)
    #  pred_array -> [n_pred, batch_size, n_route, C_0)
    pred_array = np.concatenate(pred_list, axis=1)
    return pred_array[step_idx], pred_array.shape[1], pred_array


def model_inference(sess, pred, inputs, batch_size, n_his, n_pred, step_idx, min_va_val, min_val):
    '''
    Model inference function.
    :param sess: tf.Session().
    :param pred: placeholder.
    :param inputs: instance of class Dataset, data source for inference.
    :param batch_size: int, the size of batch.
    :param n_his: int, the length of historical records for training.
    :param n_pred: int, the length of prediction.
    :param step_idx: int or list, index for prediction slice.
    :param min_va_val: np.ndarray, metric values on validation set.
    :param min_val: np.ndarray, metric values on test set.
    '''
    x_val, x_test, x_stats = inputs.get_data('val'), inputs.get_data('test'), inputs.get_stats()

    if n_his + n_pred > x_val.shape[1]:
        raise ValueError(f'ERROR: the value of n_pred "{n_pred}" exceeds the length limit.')

    y_val, len_val, x_temp = multi_pred(sess, pred, x_val, batch_size, n_his, n_pred, step_idx)
    y_val = y_val.transpose((2, 0, 1, 3))
    s1,s2,s3,s4 = y_val.shape
    y_val = y_val.reshape(s1,-1)
    y_val = z_inverse(y_val,x_stats)
    y_val = y_val.reshape(s1,s2,s3,s4)
    y_val = y_val.transpose((1, 2, 0, 3))
    evl_val = evaluation(x_val[0:len_val, step_idx + n_his, :, :], y_val, x_stats)

    # chks: indicator that reflects the relationship of values between evl_val and min_va_val.
    chks = evl_val < min_va_val
    # update the metric on test set, if model's performance got improved on the validation.
    if sum(chks):
        min_va_val[chks] = evl_val[chks]
        y_pred, len_pred, x_temp = multi_pred(sess, pred, x_test, batch_size, n_his, n_pred, step_idx)
        y_pred = y_pred.transpose((2, 0, 1, 3))
        s1,s2,s3,s4 = y_pred.shape
        y_pred = y_pred.reshape(s1,-1)
        y_pred = z_inverse(y_pred,x_stats)
        y_pred = y_pred.reshape(s1,s2,s3,s4)
        y_pred = y_pred.transpose((1, 2, 0, 3))
        evl_pred = evaluation(x_test[0:len_pred, step_idx + n_his, :, :], y_pred, x_stats)
        min_val = evl_pred
    return min_va_val, min_val


def model_test(inputs, batch_size, n_his, n_pred, inf_mode, load_path):
    '''
    Load and test saved model from the checkpoint.
    :param inputs: instance of class Dataset, data source for test.
    :param batch_size: int, the size of batch.
    :param n_his: int, the length of historical records for training.
    :param n_pred: int, the length of prediction.
    :param inf_mode: str, test mode - 'merge / multi-step test' or 'separate / single-step test'.
    :param load_path: str, the path of loaded model.
    '''
    start_time = time.time()
    model_path = tf.train.get_checkpoint_state(load_path).model_checkpoint_path

    #with  tf.device('/cpu:0'):
    test_graph = tf.Graph()

    with test_graph.as_default():
            
        with  tf.device('/cpu:0'):
            saver = tf.train.import_meta_graph(pjoin(f'{model_path}.meta'))

    with tf.Session(graph=test_graph) as test_sess:
            
        with  tf.device('/cpu:0'):
            saver.restore(test_sess, tf.train.latest_checkpoint(load_path))
            print(f'>> Loading saved model from {model_path} ...')
        #with  tf.device('/cpu:0'):
            pred = test_graph.get_collection('y_pred')

        if inf_mode == 'sep':
            # for inference mode 'sep', the type of step index is int.
            step_idx = n_pred - 1
            tmp_idx = [step_idx]
        elif inf_mode == 'merge':
            # for inference mode 'merge', the type of step index is np.ndarray.
            step_idx = tmp_idx = np.arange(3, n_pred + 1, 3) - 1
        else:
            raise ValueError(f'ERROR: test mode "{inf_mode}" is not defined.')

        x_test, x_stats = inputs.get_data('test'), inputs.get_stats()

        y_test, len_test, y_all_data = multi_pred(test_sess, pred, x_test, batch_size, n_his, n_pred, step_idx)
        y_test = y_test.transpose((2, 0, 1, 3))
        s1,s2,s3,s4 = y_test.shape
        y_test = y_test.reshape(s1,-1)
        y_test = z_inverse(y_pred,x_stats)
        y_test = y_test.reshape(s1,s2,s3,s4)
        y_test = y_test.transpose((1, 2, 0, 3))
        evl = evaluation(x_test[0:len_test, step_idx + n_his, :, :], y_test, x_stats)
        # for i in range(0,len(x_test)):
        #     y_pre = y_all_data[:,i,:,:]
        #     x_all = x_test[i,:,:,:]
        #     y_pre =z_inverse(y_pre, x_stats['mean'], x_stats['std'])
        #     x_all = z_inverse(x_all, x_stats['mean'], x_stats['std'])
        #     pd.DataFrame(np.squeeze(x_all)).to_csv('conv_x_'+str(i)+'.csv')
        #     pd.DataFrame(np.squeeze(y_pre)).to_csv('conv_y_'+str(i)+'.csv')

        for ix in tmp_idx:
            te = evl[ix - 2:ix + 1]
            print(f'Time Step {ix + 1}: MAPE {te[0]:7.3%}; MAE  {te[1]:4.3f}; RMSE {te[2]:6.3f}.')
        print(f'Model Test Time {time.time() - start_time:.3f}s')
    print('Testing model finished!')
