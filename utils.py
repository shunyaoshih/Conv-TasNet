import argparse
import os
import logging
import sys
import json
import tensorflow as tf


def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def print_num_of_trainable_parameters():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    logging.info('# of trainable parameters: %d' % total_parameters)


def read_log(log_file):
    lines = open(log_file, 'r').read().splitlines()
    best_sdr = -1000
    best_step = 0
    for line in lines:
        if 'validation SDR' in line:
            txt = line.split()
            sdr = float(txt[3])
            best_sdr = max([best_sdr, sdr])
    return best_sdr


def setup():
    parser = argparse.ArgumentParser(description='Audio Segmentation')
    parser.add_argument('-bs', '--batch_size', type=int, default=2)
    parser.add_argument('-dd', '--data_dir', type=str, default='./data')
    parser.add_argument('-ld', '--log_dir', type=str, default='logdir')
    parser.add_argument('-me', '--max_epoch', type=int, default=100)
    parser.add_argument('-mo', '--mode', type=str, default='train')
    parser.add_argument('-sr', '--sample_rate', type=int, default=8000)

    parser.add_argument('-N', '-N', type=int, default=256)
    parser.add_argument('-L', '-L', type=int, default=20)
    parser.add_argument('-B', '-B', type=int, default=256)
    parser.add_argument('-H', '-H', type=int, default=512)
    parser.add_argument('-P', '-P', type=int, default=3)
    parser.add_argument('-X', '-X', type=int, default=8)
    parser.add_argument('-R', '-R', type=int, default=4)

    args = parser.parse_args()

    args.log_file = os.path.join(args.log_dir, 'log.txt')
    args.arg_file = os.path.join(args.log_dir, 'args.json')
    args.checkpoint_path = os.path.join(args.log_dir, 'model.ckpt')

    if args.mode != 'train':
        batch_size = 1

    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)
        json.dump(vars(args), open(args.arg_file, 'w'), indent=4)

    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(args.log_file))
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)

    return args, logger
