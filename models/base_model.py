from models.layers import *
from os.path import join as pjoin
import tensorflow as tf
import keras
from keras_self_attention import SeqSelfAttention, SeqWeightedAttention


def attention_conv_layer(x):
    '''
    Temporal convolution layer.
    :param x: tensor, [batch_size, time_step, n_route, c_in].
    :param Kt: int, kernel size of temporal convolution.
    :param c_in: int, size of input channel.
    :param c_out: int, size of output channel.
    :param act_func: str, activation function.
    :return: tensor, [batch_size, time_step-Kt+1, n_route, c_out].
    '''
    _, T, n, s = x.get_shape().as_list()

    x_input = x

    # keep the original input for residual connection.

    # x_input = x_input[:, Kt - 1:T, :, :]

    _, time_step_temp, route_temp, channel_temp = x_input.get_shape().as_list()

    x_input = tf.reshape(x_input, [-1, route_temp * channel_temp, time_step_temp])
    # _, time_step_temp, route_temp, channel_temp = x_input.get_shape().as_list()
    cell = tf.keras.layers.GRUCell(route_temp)  # ,return_sequences=True)
    # x_input = tf.reshape(x_input, [-1, time_step_temp * route_temp, s])
    outputs, mid_state = tf.compat.v1.nn.dynamic_rnn(cell, x_input, dtype=tf.float32)
    x, _ = SeqSelfAttention(32, return_attention=True)(outputs)

    # x=tf.contrib.layers.l1_regularizer(0.5)(x)
    weight = tf.reshape(x, [-1, n, n])

    weight = tf.reduce_mean(weight, axis=0)
    weight = tf.nn.sigmoid(weight)

    D = tf.reduce_sum(weight, axis=1)
    # D_2 = tf.expand_dims(tf.sqrt(D),-1)
    D_2 = tf.matrix_diag(tf.sqrt(D))
    # D_2 = tf.matrix_diag(tf.pow(tf.sqrt(D), -1, name=None))
    L = tf.matmul(weight, D_2)
    L = tf.matmul(D_2, L)
    v1 = tf.Variable(tf.eye(n), name="v1")
    # L = v1 - L
    L = L
    # L=tf.nn.relu(L)

    try:
        e, v = tf.self_adjoint_eig(L)
    except:
        e, v = tf.self_adjoint_eig(weight)

    e = tf.nn.relu(e)
    v = tf.nn.relu(v)

    return e, v  # outputs


def build_model(inputs, n_his, Ks, Kt, blocks, keep_prob):
    '''
    Build the base model.
    :param inputs: placeholder.
    :param n_his: int, size of historical records for training.
    :param Ks: int, kernel size of spatial convolution.
    :param Kt: int, kernel size of temporal convolution.
    :param blocks: list, channel configs of st_conv blocks.
    :param keep_prob: placeholder.
    '''
    x = inputs[:, 0:n_his, :, :]

    # Ko>0: kernel size of temporal convolution in the output layer.
    e, v = attention_conv_layer(x)

    Ko = n_his
    # ST-Block
    flag = 0

    for i, channels in enumerate(blocks):
        if flag == 0:
            x_back = x
            l1 = 0
        x, x_back, l1 = stemGNN_block(x, Ks, Kt, channels, i, keep_prob, e, v, l1, flag, act_func='GLU',
                                      back_forecast=x_back)
        flag = 1
        Ko -= 2 * (Ks - 1)

    # Output Layer
    if Ko > 1:
        y = output_layer(x, Ko, 'output_layer')
    else:
        raise ValueError(f'ERROR: kernel size Ko must be greater than 1, but received "{Ko}".')

    tf.add_to_collection(name='copy_loss',
                         value=tf.nn.l2_loss(inputs[:, n_his - 1:n_his, :, :] - inputs[:, n_his:n_his + 1, :, :]))
    train_loss = tf.nn.l2_loss(y - inputs[:, n_his:n_his + 1, :, :]) + l1
    single_pred = y[:, 0, :, :]
    tf.add_to_collection(name='y_pred', value=single_pred)
    return train_loss, single_pred, e


def model_save(sess, global_steps, model_name, save_path):
    '''
    Save the checkpoint of trained model.
    :param sess: tf.Session().
    :param global_steps: tensor, record the global step of training in epochs.
    :param model_name: str, the name of saved model.
    :param save_path: str, the path of saved model.
    :return:
    '''
    saver = tf.train.Saver(max_to_keep=3)
    prefix_path = saver.save(sess, pjoin(save_path, model_name), global_step=global_steps)
    print(f'<< Saving model to {prefix_path} ...')
