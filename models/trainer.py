from data_loader.data_utils import gen_batch
from models.tester import model_inference
from models.base_model import build_model, model_save
from os.path import join as pjoin

import tensorflow.compat.v1 as tf
import numpy as np
import time
import logging
import pandas as pd
import os
import sys

tf.disable_eager_execution()


def print_num_of_total_parameters(output_detail=True, output_to_logging=False):
    total_parameters = 0
    parameters_string = ""

    for variable in tf.trainable_variables():

        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
        if len(shape) == 1:
            parameters_string += ("%s %d, \n" % (variable.name, variable_parameters))
        else:
            parameters_string += ("%s %s=%d, \n" % (variable.name, str(shape), variable_parameters))

    if output_to_logging:
        if output_detail:
            logging.info(parameters_string)
        logging.info("Total %d variables, %s params" % (len(tf.trainable_variables()), "{:,}".format(total_parameters)))
    else:
        if output_detail:
            print(parameters_string)
        print("Total %d variables, %s params" % (len(tf.trainable_variables()), "{:,}".format(total_parameters)))


def model_train(inputs, blocks, args, tensorboard_summary_dir, model_dir):
    '''
    Train the base model.
    :param inputs: instance of class Dataset, data source for training.
    :param blocks: list, channel configs of st_conv blocks.
    :param args: instance of class argparse, args for training.
    '''
    n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
    Ks, Kt = args.ks, args.kt
    batch_size, epoch, inf_mode, opt = args.batch_size, args.epoch, args.inf_mode, args.opt

    # Placeholder for model training
    x = tf.placeholder(tf.float32, [None, n_his + 1, n, 1], name='data_input')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # Define model loss
    train_loss, pred, e_value = build_model(x, n_his, Ks, Kt, blocks, keep_prob)
    tf.summary.scalar('train_loss', train_loss)
    copy_loss = tf.add_n(tf.get_collection('copy_loss'))
    tf.summary.scalar('copy_loss', copy_loss)

    # Learning rate settings
    global_steps = tf.Variable(0, trainable=False)
    len_train = inputs.get_len('train')
    if len_train % batch_size == 0:
        epoch_step = len_train / batch_size
    else:
        epoch_step = int(len_train / batch_size) + 1
    # Learning rate decay with rate 0.7 every 5 epochs.
    lr = tf.train.exponential_decay(args.lr, global_steps, decay_steps=5 * epoch_step, decay_rate=0.7, staircase=True)
    tf.summary.scalar('learning_rate', lr)
    step_op = tf.assign_add(global_steps, 1)
    with tf.control_dependencies([step_op]):
        if opt == 'RMSProp':
            train_op = tf.train.RMSPropOptimizer(lr).minimize(train_loss)
        elif opt == 'ADAM':
            train_op = tf.train.AdamOptimizer(lr).minimize(train_loss)
        else:
            raise ValueError(f'ERROR: optimizer "{opt}" is not defined.')

    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        if not os.path.exists(tensorboard_summary_dir):
            os.makedirs(tensorboard_summary_dir)
        writer = tf.summary.FileWriter(pjoin(tensorboard_summary_dir), sess.graph)
        sess.run(tf.global_variables_initializer())
        print("-----------------------------------------" * 2)
        print_num_of_total_parameters()
        print("---------------**********-------------------" * 2)

        if inf_mode == 'sep':
            # for inference mode 'sep', the type of step index is int.
            step_idx = n_pred - 1
            tmp_idx = [step_idx]
            min_val = min_va_val = np.array([4e1, 1e5, 1e5])
        elif inf_mode == 'merge':
            # for inference mode 'merge', the type of step index is np.ndarray.
            step_idx = tmp_idx = np.arange(1, n_pred+1, 1) - 1
            min_val = min_va_val = np.array([4e1, 1e5, 1e5] * len(step_idx))
        else:
            raise ValueError(f'ERROR: test mode "{inf_mode}" is not defined.')

        minimum_mape = sys.float_info.max

        for i in range(epoch):
            start_time = time.time()
            for j, x_batch in enumerate(
                    gen_batch(inputs.get_data('train'), batch_size, dynamic_batch=True, shuffle=True)):
                summary, _ = sess.run([merged, train_op], feed_dict={x: x_batch[:, 0:n_his + 1, :, :], keep_prob: 1.0})
                writer.add_summary(summary, i * epoch_step + j)
                if j % 50 == 0:
                    loss_value = \
                        sess.run([train_loss, copy_loss],
                                 feed_dict={x: x_batch[:, 0:n_his + 1, :, :], keep_prob: 1.0})
                    print(f'Epoch {i:2d}, Step {j:3d}: [{loss_value[0]:.3f}, {loss_value[1]:.3f}]')
            print(f'Epoch {i:2d} Training Time {time.time() - start_time:.3f}s')

            # e_value_1 = sess.run([e_value], feed_dict={x: x_batch[:, 0:n_his + 1, :, :], keep_prob: 1.0})

            # pd.DataFrame(e_value_1).to_csv("e_value" + str(i) + ".csv")
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)


            if (i + 1) % args.save == 0:
                start_time = time.time()
                
                evl, min_val = \
                         model_inference(sess, pred, inputs, batch_size, n_his, n_pred, step_idx, min_va_val, min_val)

                # for ix in tmp_idx:
                #     va, te = min_va_val[ix - 2:ix + 1], min_val[ix - 2:ix + 1]
                #     print(f'Time Step {ix + 1}: '
                #           f'MAPE {va[0]:7.3%}, {te[0]:7.3%}; '
                #           f'MAE  {va[1]:4.3f}, {te[1]:4.3f}; '
                #           f'RMSE {va[2]:6.3f}, {te[2]:6.3f}.')
                amape=[]
                amae=[]
                armse=[]
                for ix in tmp_idx:
                    te = evl[ix*3:ix*3 + 3]
                    amape.append(te[0])
                    amae.append(te[1])
                    armse.append(te[2])
                    print(f'Time Step {ix + 1}: MAPE {te[0]:7.3%}; MAE  {te[1]:4.3f}; RMSE {te[2]:6.3f}.')
                
                print(f'Results: MAPE {np.mean(amape):7.3%}; MAE  {np.mean(amae):4.3f}; RMSE {np.mean(armse):6.3f}.')

                print(f'Epoch {i:2d} Inference Time {time.time() - start_time:.3f}s')

                

                model_save(sess, global_steps, 'StemGNN', model_dir)

                if evl[1] <= minimum_mape:
                    minimum_mape = evl[1]
                    best_model_dir_name = pjoin(model_dir, 'best')
                    if not os.path.exists(best_model_dir_name):
                        os.makedirs(best_model_dir_name)
                    model_save(sess, global_steps, 'StemGNN', best_model_dir_name)

        writer.close()
    print('Training model finished!')
