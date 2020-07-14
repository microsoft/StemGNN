import tensorflow as tf
from keras_self_attention import SeqSelfAttention, SeqWeightedAttention


def gconv_fft_cnn_0221(x, theta, Ks, c_in, c_out, e):
    '''
    Spectral-based graph convolution function.
    :param x: tensor, [batch_size, n_route, c_in].spatio_conv_layer_fft_2
    :param theta: tensor, [Ks*c_in, c_out], trainable kernel parameters.
    :param Ks: int, kernel size of graph convolution.
    :param c_in: int, size of input channel.
    :param c_out: int, size of output channel.
    :return: tensor, [batch_size, n_route, c_out].
    '''
    # graph kernel: tensor, [n_route, Ks*n_route]
    # if Ks != 1:
    #    kernel = tf.get_collection('graph_kernel_2')[0]
    # if Ks == 1:
    #    kernel = tf.get_collection('graph_V')[0]
    kernel = tf.expand_dims(e, -1)
    kernel = tf.matrix_diag(e)
    n = tf.shape(kernel)[0]
    # [batch_size, time_step, n_route, c_in]
    batch_size, time_step, n_route, c_in = x.get_shape().as_list()
    # x -> [batch_size, c_in, n_route] -> [batch_size*c_in, n_route]
    # x_tmp = tf.reshape(x,[-1,c_in])
    #
    x_tmp = tf.reshape(tf.transpose(x, [0, 1, 3, 2]), [-1, n])
    #real_kernel = tf.multiply(theta, kernel) +theta
    real_kernel = tf.matmul(theta, kernel) #+theta
    x_mul = tf.matmul(x_tmp, real_kernel)
    x_gconv = tf.transpose(tf.reshape(x_mul, [-1, c_out, n_route]),[0,2,1])
    # x_mul = x_tmp * ker -> [batch_size*c_in, Ks*n_route] -> [batch_size, c_in, Ks, n_route]
    # x_mul = tf.reshape(tf.matmul(x_tmp, kernel), [-1, time_step, c_in, n_route])
    # x_mul = tf.transpose(x_mul, [0, 1, 3, 2])
    # theta = tf.get_variable('theta_input', shape=[c_in, c_in, c_in, 1], dtype=tf.float32)
    # tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(theta))

    # x_input = tf.nn.conv2d(x_mul, theta, strides=[1, 1, 1, 1], padding='SAME')

    # x_gconv = x_input  # tf.reshape(tf.matmul(x_input, theta), [-1, n, c_out])

    return x_gconv


def graph_fft(x, v, flag=True):
    '''

    [batch_size, time_step, n_route, c_in].

    :return: tensor, x.
    '''

    _, T, n, c_in = x.get_shape().as_list()
    if flag:
        U = tf.transpose(v)  # tf.get_collection('graph_U_T')[0]  # (228*228 n*n)
    else:
        U = v  # tf.get_collection('graph_U')[0]
    #x_tmp = tf.reshape(tf.transpose(x, [2, 0, 1, 3]), [n, -1])
    x = tf.matmul(U, x)  # .reshape()
    # x = tf.multiply( U, x_tmp)
    #x = tf.reshape(x, [-1, T, n, c_in])
    return x


def layer_norm(x, scope):
    '''
    Layer normalization function.
    :param x: tensor, [batch_size, time_step, n_route, channel].
    :param scope: str, variable scope.
    :return: tensor, [batch_size, time_step, n_route, channel].
    '''
    _, _, N, C = x.get_shape().as_list()
    mu, sigma = tf.nn.moments(x, axes=[2, 3], keep_dims=True)

    with tf.variable_scope(scope):
        gamma = tf.get_variable('gamma', initializer=tf.ones([1, 1, N, C]))
        beta = tf.get_variable('beta', initializer=tf.zeros([1, 1, N, C]))
        _x = (x - mu) / tf.sqrt(sigma + 1e-6) * gamma + beta
    return _x


def temporal_conv_layer(x, Kt, c_in, c_out, act_func='relu'):
    '''
    Temporal convolution layer.
    :param x: tensor, [batch_size, time_step, n_route, c_in].
    :param Kt: int, kernel size of temporal convolution.
    :param c_in: int, size of input channel.
    :param c_out: int, size of output channel.
    :param act_func: str, activation function.
    :return: tensor, [batch_size, time_step-Kt+1, n_route, c_out].
    '''
    _, T, n, _ = x.get_shape().as_list()

    # w_input = tf.get_variable('wt_input', shape=[32, 32, c_in, 1], dtype=tf.float32)
    # tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w_input))
    # x_input = tf.nn.conv2d(x, w_input, strides=[1, 1, 1, 1], padding='SAME')

    if c_in > c_out:
        w_input = tf.get_variable('wt_input', shape=[1, 1, c_in, c_out], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w_input))
        x_input = tf.nn.conv2d(x, w_input, strides=[1, 1, 1, 1], padding='SAME')
    elif c_in < c_out:
        # if the size of input channel is less than the output,
        # padding x to the same size of output channel.
        # Note, _.get_shape() cannot convert a partially known TensorShape to a Tensor.
        x_input = tf.concat([x, tf.zeros([tf.shape(x)[0], T, n, c_out - c_in])], axis=3)
    else:
        x_input = x

    # keep the original input for residual connection.
    _, time_step_temp, route_temp, channel_temp = x_input.get_shape().as_list()
    x_input = x_input[:, Kt - 1:T, :, :]

    if act_func == 'GLU':
        # gated liner unit
        wt = tf.get_variable(name='wt', shape=[Kt, 1, c_in, 2 * c_out], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(wt))
        bt = tf.get_variable(name='bt', initializer=tf.zeros([2 * c_out]), dtype=tf.float32)
        x_conv = tf.nn.conv2d(x, wt, strides=[1, 1, 1, 1], padding='VALID') + bt
        return (x_conv[:, :, :, 0:c_out] + x_input) * tf.nn.sigmoid(x_conv[:, :, :, -c_out:])
    elif act_func == 'GRU':
        _, _, _, channel_temp = x_input.get_shape().as_list()
        cell = tf.keras.layers.GRUCell(channel_temp)  # ,return_sequences=True)
        x_input = tf.reshape(x_input, [-1, T - Kt + 1, channel_temp])
        outputs, _ = tf.nn.dynamic_rnn(cell, x_input, dtype=tf.float32)
        outputs = tf.reshape(outputs, [-1, T - Kt + 1, n, channel_temp])
        return outputs
    elif act_func == 'GRU_shape':

        cell = tf.keras.layers.GRUCell(channel_temp)  # ,return_sequences=True)
        x_input = tf.reshape(x_input, [-1, T - Kt + 1, channel_temp])  # route_temp*
        outputs, _ = tf.nn.dynamic_rnn(cell, x_input, dtype=tf.float32)
        outputs = tf.reshape(outputs, [-1, time_step_temp, route_temp, channel_temp])
        return outputs
    else:
        wt = tf.get_variable(name='wt', shape=[Kt, 1, c_in, c_out], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(wt))
        bt = tf.get_variable(name='bt', initializer=tf.zeros([c_out]), dtype=tf.float32)
        x_conv = tf.nn.conv2d(x, wt, strides=[1, 1, 1, 1], padding='VALID') + bt
        if act_func == 'linear':
            return x_conv
        elif act_func == 'sigmoid':
            return tf.nn.sigmoid(x_conv)
        elif act_func == 'relu':
            return tf.nn.relu(x_conv + x_input)
        else:
            raise ValueError(f'ERROR: activation function "{act_func}" is not defined.')


def temporal_conv_layer_imag(x, Kt, c_in, c_out, act_func='relu'):
    '''
    Temporal convolution layer.
    :param x: tensor, [batch_size, time_step, n_route, c_in].
    :param Kt: int, kernel size of temporal convolution.
    :param c_in: int, size of input channel.
    :param c_out: int, size of output channel.
    :param act_func: str, activation function.
    :return: tensor, [batch_size, time_step-Kt+1, n_route, c_out].
    '''
    _, T, n, _ = x.get_shape().as_list()

    if c_in > c_out:
        w_imag_input = tf.get_variable('wt_imag_input', shape=[1, 1, c_in, c_out], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w_imag_input))
        x_input = tf.nn.conv2d(x, w_imag_input, strides=[1, 1, 1, 1], padding='SAME')
    elif c_in < c_out:
        # if the size of input channel is less than the output,
        # padding x to the same size of output channel.
        # Note, _.get_shape() cannot convert a partially known TensorShape to a Tensor.
        x_input = tf.concat([x, tf.zeros([tf.shape(x)[0], T, n, c_out - c_in])], axis=3)
    else:
        x_input = x

    # keep the original input for residual connection.
    x_input = x_input[:, Kt - 1:T, :, :]
    _, _, _, channel_temp = x_input.get_shape().as_list()

    if act_func == 'GLU':
        # gated liner unit
        wt_imag = tf.get_variable(name='wt_imag', shape=[Kt, 1, c_in, 2 * c_out], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(wt_imag))
        bt_imag = tf.get_variable(name='bt_imag', initializer=tf.zeros([2 * c_out]), dtype=tf.float32)
        x_conv = tf.nn.conv2d(x, wt_imag, strides=[1, 1, 1, 1], padding='VALID') + bt_imag
        return (x_conv[:, :, :, 0:c_out] + x_input) * tf.nn.sigmoid(x_conv[:, :, :, -c_out:])
    elif act_func == 'GRU':
        cell = tf.keras.layers.GRUCell(channel_temp)  # ,return_sequences=True)
        x_input = tf.reshape(x_input, [-1, T - Kt + 1, channel_temp])
        outputs, _ = tf.compat.v1.nn.dynamic_rnn(cell, x_input, dtype=tf.float32)
        outputs = tf.reshape(outputs, [-1, T - Kt + 1, n, channel_temp])
        return outputs
    else:
        wt_imag = tf.get_variable(name='wt_imag', shape=[Kt, 1, c_in, c_out], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(wt_imag))
        bt_imag = tf.get_variable(name='bt_imag', initializer=tf.zeros([c_out]), dtype=tf.float32)
        x_conv = tf.nn.conv2d(x, wt_imag, strides=[1, 1, 1, 1], padding='VALID') + bt_imag
        if act_func == 'linear':
            return x_conv
        elif act_func == 'sigmoid':
            return tf.nn.sigmoid(x_conv)
        elif act_func == 'relu':
            return tf.nn.relu(x_conv + x_input)
        else:
            raise ValueError(f'ERROR: activation function "{act_func}" is not defined.')


def temporal_conv_layer_input(x, Kt, c_in, c_out, act_func='relu', type='x'):
    '''
    Temporal convolution layer.
    :param x: tensor, [batch_size, time_step, n_route, c_in].
    :param Kt: int, kernel size of temporal convolution.
    :param c_in: int, size of input channel.
    :param c_out: int, size of output channel.
    :param act_func: str, activation function.
    :return: tensor, [batch_size, time_step-Kt+1, n_route, c_out].
    '''
    _, T, n, _ = x.get_shape().as_list()

    # w_input = tf.get_variable('wt_input', shape=[32, 32, c_in, 1], dtype=tf.float32)
    # tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w_input))
    # x_input = tf.nn.conv2d(x, w_input, strides=[1, 1, 1, 1], padding='SAME')

    if c_in > c_out:
        w_input = tf.get_variable('wt_input3' + type, shape=[1, 1, c_in, c_out], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w_input))
        x_input = tf.nn.conv2d(x, w_input, strides=[1, 1, 1, 1], padding='SAME')
    elif c_in < c_out:
        # if the size of input channel is less than the output,
        # padding x to the same size of output channel.
        # Note, _.get_shape() cannot convert a partially known TensorShape to a Tensor.
        x_input = tf.concat([x, tf.zeros([tf.shape(x)[0], T, n, c_out - c_in])], axis=3)
    else:
        x_input = x

    # keep the original input for residual connection.
    _, time_step_temp, route_temp, channel_temp = x_input.get_shape().as_list()
    x_input = x_input[:, Kt - 1:T, :, :]

    if act_func == 'GLU':
        # gated liner unit
        wt = tf.get_variable(name='wt', shape=[Kt, 1, c_in, 2 * c_out], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(wt))
        bt = tf.get_variable(name='bt', initializer=tf.zeros([2 * c_out]), dtype=tf.float32)
        x_conv = tf.nn.conv2d(x, wt, strides=[1, 1, 1, 1], padding='VALID') + bt
        return (x_conv[:, :, :, 0:c_out] + x_input) * tf.nn.sigmoid(x_conv[:, :, :, -c_out:])
    elif act_func == 'GRU':
        _, _, _, channel_temp = x_input.get_shape().as_list()
        cell = tf.keras.layers.GRUCell(channel_temp)  # ,return_sequences=True)
        x_input = tf.reshape(x_input, [-1, T - Kt + 1, channel_temp])
        outputs, _ = tf.nn.dynamic_rnn(cell, x_input, dtype=tf.float32)
        outputs = tf.reshape(outputs, [-1, T - Kt + 1, n, channel_temp])
        return outputs
    elif act_func == 'GRU_shape':

        cell = tf.keras.layers.GRUCell(channel_temp)  # ,return_sequences=True)
        x_input = tf.reshape(x_input, [-1, T - Kt + 1, channel_temp])  # route_temp*
        outputs, _ = tf.nn.dynamic_rnn(cell, x_input, dtype=tf.float32)
        outputs = tf.reshape(outputs, [-1, time_step_temp, route_temp, channel_temp])
        return outputs
    else:
        wt = tf.get_variable(name='wt_input2' + type, shape=[Kt, 1, c_in, c_out], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(wt))
        bt = tf.get_variable(name='bt_input2' + type, initializer=tf.zeros([c_out]), dtype=tf.float32)
        x_conv = tf.nn.conv2d(x, wt, strides=[1, 1, 1, 1], padding='VALID') + bt
        if act_func == 'linear':
            return x_conv
        elif act_func == 'sigmoid':
            return tf.nn.sigmoid(x_conv)
        elif act_func == 'relu':
            return tf.nn.relu(x_conv + x_input)
        else:
            raise ValueError(f'ERROR: activation function "{act_func}" is not defined.')


def fc(x, type='fore'):
    '''

    :param x: tensor, [batch_size, time_step, n_route, c_in].
    :param Kt: int, kernel size of temporal convolution.
    :param c_in: int, size of input channel.
    :param c_out: int, size of output channel.
    :param act_func: str, activation function.
    :return: tensor, [batch_size, time_step-Kt+1, n_route, c_out].
    '''
    # _, T, n, _ = x.get_shape().as_list()
    # keep the original input for residual connection.
    _, time_step_temp, route_temp, channel_temp = x.get_shape().as_list()
    # x_input = x_input[:, Kt - 1:T, :, :]
    x_tmp = tf.reshape(x, [-1, channel_temp])
    # x_tmp = tf.reshape(tf.transpose(x, [0, 2, 3, 1]), [-1, time_step_temp])
    # [time_step_temp,T-Kt+1]
    wt = tf.get_variable(name='wt_' + type, shape=[channel_temp, channel_temp], dtype=tf.float32)
    tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(wt))
    bt = tf.get_variable(name='bt_' + type, initializer=tf.zeros([channel_temp]), dtype=tf.float32)
    hidden = tf.sigmoid(tf.add(tf.matmul(x_tmp, wt), bt))
    #out = tf.nn.softmax(hidden)
    out = tf.nn.sigmoid(hidden)
    outputs = tf.reshape(out, [-1, time_step_temp, route_temp, channel_temp])
    return outputs


def fore_auto(x, Kt, c_in, c_out):
    '''

    :param x: tensor, [batch_size, time_step, n_route, c_in].
    :param Kt: int, kernel size of temporal convolution.
    :param c_in: int, size of input channel.
    :param c_out: int, size of output channel.
    :param act_func: str, activation function.
    :return: tensor, [batch_size, time_step-Kt+1, n_route, c_out].
    '''
    _, T, n, _ = x.get_shape().as_list()

    # w_input = tf.get_variable('wt_input', shape=[32, 32, c_in, 1], dtype=tf.float32)
    # tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w_input))
    # x_input = tf.nn.conv2d(x, w_input, strides=[1, 1, 1, 1], padding='SAME')

    if c_in > c_out:
        w_input = tf.get_variable('wt_input_fore', shape=[1, 1, c_in, c_out], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w_input))
        x_input = tf.nn.conv2d(x, w_input, strides=[1, 1, 1, 1], padding='SAME')
    elif c_in < c_out:
        # if the size of input channel is less than the output,
        # padding x to the same size of output channel.
        # Note, _.get_shape() cannot convert a partially known TensorShape to a Tensor.
        x_input = tf.concat([x, tf.zeros([tf.shape(x)[0], T, n, c_out - c_in])], axis=3)
    else:
        x_input = x

    # keep the original input for residual connection.
    _, time_step_temp, route_temp, channel_temp = x.get_shape().as_list()
    x_input = x_input[:, Kt - 1:T, :, :]

    x_tmp = tf.reshape(tf.transpose(x, [0, 2, 3, 1]), [-1, time_step_temp])
    # [time_step_temp,T-Kt+1]
    wt = tf.get_variable(name='wt_en', shape=[time_step_temp, T - Kt + 1], dtype=tf.float32)
    tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(wt))
    bt = tf.get_variable(name='bt_en', initializer=tf.zeros([T - Kt + 1]), dtype=tf.float32)
    hidden = tf.sigmoid(tf.add(tf.matmul(x_tmp, wt), bt))
    hidden = tf.reshape(hidden, [-1, channel_temp])
    # x_tmp = tf.reshape(tf.transpose(x, [0, 2, 3, 1]), [-1, time_step_temp])
    # [time_step_temp,T-Kt+1]
    wt_de = tf.get_variable(name='wt_de', shape=[channel_temp, c_out], dtype=tf.float32)
    tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(wt_de))
    bt_de = tf.get_variable(name='bt_de', initializer=tf.zeros([c_out]), dtype=tf.float32)
    out = tf.nn.softmax(tf.add(tf.matmul(hidden, wt_de), bt_de))
    outputs = tf.reshape(out, [-1, T - Kt + 1, route_temp, c_out])

    # x_conv = tf.nn.conv2d(x, wt, strides=[1, 1, 1, 1], padding='VALID') + bt

    return outputs


def spatio_conv_layer_fft_0221(x, Ks, c_in, c_out, e):
    '''

    :param x: tensor, [batch_size, time_step, n_route, c_in].
    :param Ks: int, kernel size of spatial convolution.
    :param c_in: int, size of input channel.
    :param c_out: int, size of output channel.
    :return: tensor, [batch_size, time_step, n_route, c_out].
    '''
    _, T, n, _ = x.get_shape().as_list()

    Ks = 1

    if c_in > c_out:
        # bottleneck down-sampling
        w_input = tf.get_variable('ws_input_2', shape=[1, 1, c_in, c_out], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w_input))
        x_input = tf.nn.conv2d(x, w_input, strides=[1, 1, 1, 1], padding='SAME')
    elif c_in < c_out:
        # if the size of input channel is less than the output,
        # padding x to the same size of output channel.
        # Note, _.get_shape() cannot convert a partially known TensorShape to a Tensor.
        x_input = tf.concat([x, tf.zeros([tf.shape(x)[0], T, n, c_out - c_in])], axis=3)
    else:
        x_input = x

    # shape=[c_in, c_in, c_in, 1] Ks * c_in, c_out
    # ws = tf.get_variable(name='ws', shape=[Ks * c_in, c_out], dtype=tf.float32)
    ws = tf.get_variable(name='ws_2', shape=[n, n], dtype=tf.float32)
    tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(ws))
    variable_summaries(ws, 'theta')
    bs = tf.get_variable(name='bs_2', initializer=tf.zeros([c_out]), dtype=tf.float32)
    # x -> [batch_size*time_step, n_route, c_in] -> [batch_size*time_step, n_route, c_out]

    GF = x
    x_gc = gconv_fft_cnn_0221(GF, ws, Ks, c_in, c_out, e) + bs

    # x_gconv = gconv_fft(tf.reshape(GF, [-1, n, c_out]), ws, Ks, c_in, c_out) + bs

    # x_gconv = gconv_fft_only_mul(tf.reshape(GF, [-1, n, c_out]), ws, Ks, c_in, c_out) + bs

    x_gc = tf.reshape(x_gc, [-1, T, n, c_out])
    # x_g -> [batch_size, time_step, n_route, c_out]

    # x_gc = tf.reshape(x_gconv, [-1, T, n, c_out])

    # return tf.nn.relu(GF[:, :, :, 0:c_out] + x_input)
    return x_gc  # tf.nn.relu(x_gc[:, :, :, 0:c_out]+ x_input )


def stemGNN_block(x, Ks, Kt, channels, scope, keep_prob, e, v, l1, flag=0, act_func='GLU', back_forecast=None):
    '''
    GFFT-fft-GRU-ifft-attention-gconv-iGfft-GLU

    :param x: tensor, batch_size, time_step, n_route, c_in].
    :param Ks: int, kernel size of spatial convolution.
    :param Kt: int, kernel size of temporal convolution.
    :param channels: list, channel configs of a single st_conv block.
    :param scope: str, variable scope.
    :param keep_prob: placeholder, prob of dropout.
    :param act_func: str, activation function.
    :return: tensor, [batch_size, time_step, n_route, c_out].
    '''
    c_si, c_t, c_oo = channels

    len, T, n, c_in = x.get_shape().as_list()

    with tf.variable_scope(f'stn_block_{scope}_in'):

        x_input = temporal_conv_layer_input(x, Kt, c_si, c_t, 'relu')
        # flag = False
        if flag == 0:
            # flag = True
            back_forecast = x_input
        else:
            back_forecast = fore_auto(back_forecast, Kt, c_in, c_t)

        GF = graph_fft(x, v, True)
        x = GF

        #x = tf.reshape(tf.transpose(x, [0, 2, 3, 1]), [-1, T])
        #x = tf.spectral.fft(tf.cast(x, dtype=tf.complex64))
        x = tf.real(x)
        x_imag = tf.imag(x)
        
        #x =  tf.transpose(tf.reshape(x,[-1,n,c_in,T]),[0,3,1,2])
        #x_imag =  tf.transpose(tf.reshape(x_imag,[-1,n,c_in,T]),[0,3,1,2])
        # c_in = c_t
        _, time_step_temp, route_temp, channel_temp = x.get_shape().as_list()
        x = temporal_conv_layer(x, Kt, c_si, c_t, 'GLU')
        x_imag = temporal_conv_layer_imag(x_imag, Kt, c_si, c_t, 'GLU')
        _, T, n, cc = x.get_shape().as_list()
        #x = tf.reshape(tf.transpose(x, [0, 2, 3, 1]), [-1, T])
        #x_imag = tf.reshape(tf.transpose(x_imag, [0, 2, 3, 1]), [-1, T])
        x = tf.to_float(tf.spectral.ifft((tf.complex(x, x_imag))))
        #x =  tf.transpose(tf.reshape(x,[-1,n,cc,T]),[0,3,1,2])

        #x = tf.reshape(x,[-1,n,c_in])

        # g_conv
        _, _, _, c_fft = x.get_shape().as_list()

        x = spatio_conv_layer_fft_0221(x, Ks, c_fft, c_fft, e)

        # x = spatio_conv_layer_fft_0222(x, Ks, c_fft, c_fft,e)

        GF = graph_fft(x, v, False)
        x = GF
        #
        # if flag == 0:
        #     # flag = True
        #     back_forecast = x_input
        # else:
        #     back_forecast = fore_auto(back_forecast, Kt, c_in, c_t)

        back_cast = fc(x, 'back')
        x = back_cast
        x = tf.nn.relu(x[:, :, :, 0:c_fft] + x_input, 'relu_test')

        # x= attention_conv_layer(x)
        if flag == 0:
            # back_forecast=x_input
            x = tf.nn.relu(x[:, :, :, 0:c_fft] + x_input, 'relu_test')
            l1 = tf.nn.l2_loss(x_input - back_cast)  # x[:, :, :, 0:c_fft])
        else:
            # back_forecast=fore_auto(back_forecast, Kt, c_in, c_t)
            x = tf.nn.relu(x[:, :, :, 0:c_fft] + back_forecast + x_input, 'relu_test')
            # l1 = l1 + tf.nn.l2_loss(back_forecast - x[:, :, :, 0:c_fft])

        x_t = x  # tf.add(x_t1, 0.5 * x_t2)

    with tf.variable_scope(f'stn_block_{scope}_out'):
        x_o = temporal_conv_layer(x_t, Kt, c_t, c_oo)
        back_cast_o = temporal_conv_layer_imag(back_forecast, Kt, c_t, c_oo)
    x_ln = layer_norm(x_o, f'layer_norm_{scope}')
    fore_cast_ln = layer_norm(back_cast_o, f'layer_norm_{scope}_back')
    return tf.nn.dropout(x_ln, keep_prob), tf.nn.dropout(fore_cast_ln, keep_prob), l1


def fully_con_layer(x, n, channel, scope):
    '''
    Fully connected layer: maps multi-channels to one.
    :param x: tensor, [batch_size, 1, n_route, channel].
    :param n: int, number of route / size of graph.
    :param channel: channel size of input x.
    :param scope: str, variable scope.
    :return: tensor, [batch_size, 1, n_route, 1].
    '''
    w = tf.get_variable(name=f'w_{scope}', shape=[1, 1, channel, 1], dtype=tf.float32)
    tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w))
    b = tf.get_variable(name=f'b_{scope}', initializer=tf.zeros([n, 1]), dtype=tf.float32)
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME') + b


def output_layer(x, T, scope, act_func='GLU'):
    '''
    Output layer: temporal convolution layers attach with one fully connected layer,
    which map outputs of the last st_conv block to a single-step prediction.
    :param x: tensor, [batch_size, time_step, n_route, channel].
    :param T: int, kernel size of temporal convolution.
    :param scope: str, variable scope.
    :param act_func: str, activation function.
    :return: tensor, [batch_size, 1, n_route, 1].
    '''
    _, _, n, channel = x.get_shape().as_list()

    # maps multi-steps to one.
    with tf.variable_scope(f'{scope}_in'):
        x_i = temporal_conv_layer(x, T, channel, channel, act_func=act_func)
    # x_i=tf.slice(x, [0,0, 0, 0], [-1, 1,-1,-1])
    x_ln = layer_norm(x_i, f'layer_norm_{scope}')
    with tf.variable_scope(f'{scope}_out'):
        x_o = temporal_conv_layer(x_ln, 1, channel, channel, act_func='sigmoid')
    # maps multi-channels to one.
    x_fc = fully_con_layer(x_o, n, channel, scope)
    return x_fc


def variable_summaries(var, v_name):
    '''
    Attach summaries to a Tensor (for TensorBoard visualization).
    Ref: https://zhuanlan.zhihu.com/p/33178205
    :param var: tf.Variable().
    :param v_name: str, name of the variable.
    '''
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar(f'mean_{v_name}', mean)

        with tf.name_scope(f'stddev_{v_name}'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar(f'stddev_{v_name}', stddev)

        tf.summary.scalar(f'max_{v_name}', tf.reduce_max(var))
        tf.summary.scalar(f'min_{v_name}', tf.reduce_min(var))

        tf.summary.histogram(f'histogram_{v_name}', var)
