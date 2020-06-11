import os
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from os.path import join as pjoin

import tensorflow as tf
from datetime import datetime

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.Session(config=config)

from utils.math_graph import *
from data_loader.data_utils import *
from models.trainer import model_train
from models.tester import model_test

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_route', type=int, default=140)
parser.add_argument('--n_his', type=int, default=12)
parser.add_argument('--n_pred', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--save', type=int, default=10)
parser.add_argument('--ks', type=int, default=3)
parser.add_argument('--kt', type=int, default=3)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--opt', type=str, default='RMSProp')
parser.add_argument('--graph', type=str, default='default')
parser.add_argument('--inf_mode', type=str, default='merge')
parser.add_argument('--dataset', type=str, default='./dataset/ECG_data.csv')
parser.add_argument('--output_dir', type=str, default='./output/ECG_data')
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--evaluate', type=bool, default=True)

args = parser.parse_args()
print(f'Training configs: {args}')

n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
Ks, Kt = args.ks, args.kt
# blocks: settings of channel size in Spe-Seq Cell
blocks = [[1, 32, 64], [64, 32, 128]]
output_dir = args.output_dir

# Data Preprocessing
data_file = args.dataset

DATA = data_gen(data_file, n, n_his + n_pred)
print(f'>> Loading dataset ')

SEED = 1
if __name__ == '__main__':
    tf.compat.v1.set_random_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    if args.train :
        before_train = datetime.now().timestamp()
        model_train(DATA, blocks, args, tensorboard_summary_dir=pjoin(output_dir, 'tensorboard'),
                    model_dir=pjoin(output_dir, 'model'))
        after_train = datetime.now().timestamp()
    if args.evaluate :
        before_evaluation = datetime.now().timestamp()
        model_test(DATA, DATA.get_len('test'), n_his, n_pred, args.inf_mode, load_path=pjoin(output_dir, 'model'))
        after_evaluation = datetime.now().timestamp()
    print('Duration Overview:')
    if args.train :
        print(f'Training took {(after_train - before_train)/60} minutes')
    if args.evaluate :
        print(f'Evaluation took {(after_evaluation - before_evaluation)/60} minutes')
