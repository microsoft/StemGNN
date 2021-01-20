import os
import random
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from os.path import join as pjoin
from datetime import datetime
from utils.math_graph import *
from models.trainer import train
from models.tester import model_test
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--window_size', type=int, default=12)
parser.add_argument('--horizon', type=int, default=3)
parser.add_argument('--train_length', type=float, default=7)
parser.add_argument('--valid_length', type=float, default=2)
parser.add_argument('--test_length', type=float, default=1)
parser.add_argument('--model_config', type=str, default='./utils/model_config.json')
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--save', type=int, default=5)
parser.add_argument('--ks', type=int, default=3)
parser.add_argument('--kt', type=int, default=3)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--opt', type=str, default='RMSProp')
parser.add_argument('--graph', type=str, default='default')
parser.add_argument('--inf_mode', type=str, default='merge')
parser.add_argument('--dataset', type=str, default='ECG_data')
parser.add_argument('--scalar', type=str, default='zscore')
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--evaluate', type=bool, default=True)
parser.add_argument('--multi_layer', type=int, default=5)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--stack_count', type=int, default=2)
parser.add_argument('--validate_freq', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--exponential_decay_step', type=int, default=5)
parser.add_argument('--decay_rate', type=float, default=0.5)
parser.add_argument('--dropout_rate', type=float, default=0.5)
parser.add_argument('--leakyrelu_rate', type=int, default=0.2)
parser.add_argument('--optimizer', type=str, default='RMSProp')
parser.add_argument('--norm_method', type=str, default='RMSProp')
parser.add_argument('--early_stop', type=bool, default=False)

args = parser.parse_args()
print(f'Training configs: {args}')
data_file = os.path.join('dataset', args.dataset + '.csv')
result_file = os.path.join('output', args.dataset)
if not os.path.exists(result_file):
    os.makedirs(result_file)
data = pd.read_csv(data_file).values
train_ratio = args.train_length / (args.train_length + args.valid_length + args.test_length)
valid_ratio = args.valid_length / (args.train_length + args.valid_length + args.test_length)
test_ratio = 1 - train_ratio - valid_ratio
train_data = data[:int(train_ratio * len(data))]
valid_data = data[int(train_ratio * len(data)):int((train_ratio + valid_ratio) * len(data))]
test_data = data[int((train_ratio + valid_ratio) * len(data)):]

torch.manual_seed(0)
if __name__ == '__main__':
    if args.train:
        before_train = datetime.now().timestamp()
        train(train_data, valid_data, args, result_file)
        after_train = datetime.now().timestamp()
    if args.evaluate:
        before_evaluation = datetime.now().timestamp()
        model_test(DATA, DATA.get_len('test'), n_his, n_pred, args.inf_mode,
                   load_path=pjoin(output_dir, 'model', 'best'))
        after_evaluation = datetime.now().timestamp()
    print('Duration Overview:')
    if args.train:
        print(f'Training took {(after_train - before_train) / 60} minutes')
    if args.evaluate:
        print(f'Evaluation took {(after_evaluation - before_evaluation) / 60} minutes')
