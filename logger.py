import tensorflow as tf
import pathlib
from datetime import datetime
import os


class Logger(object):
    def __init__(self, log_dir):
        self.writer = tf.summary.FileWriter(log_dir)

    def log_scalar(self, tag, value, step):

        summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                     simple_value=value)])
        self.writer.add_summary(summary, step)


def setup_logger(args):
    if args.logger:
        if args.alg == 'nns':
            logger_nns = Logger(pathlib.Path(os.getcwd()) / 'train_logs' / f'RL-agent-{datetime.now()}')
            logger_heft = None
        elif args.alg == 'heft':
            logger_nns = None
            logger_heft = Logger(pathlib.Path(os.getcwd()) / 'train_logs' / f'RL-agent-{datetime.now()}')
        elif args.alg == 'compare':
            base_dir = pathlib.Path(os.getcwd()) / 'train_logs' / f'RL-agent-{datetime.now()}'
            logger_nns = Logger(log_dir=base_dir / 'nns')
            logger_heft = Logger(log_dir=base_dir / 'heft')
    else:
        logger_nns = None
        logger_heft = None

    return logger_nns, logger_heft

