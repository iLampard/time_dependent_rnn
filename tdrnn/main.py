""" A script to run the model """

import os
import sys

CUR_PATH = os.path.abspath(os.path.dirname(__file__))
ROOT_PATH = os.path.split(CUR_PATH)[0]
sys.path.append(ROOT_PATH)

from absl import app
from absl import flags
from tdrnn.data_loader import BaseLoader
from tdrnn.model_runner import ModelRunner

FLAGS = flags.FLAGS

# Data input params
flags.DEFINE_string('data_set', 'data_retweet', 'Source data set for training')
flags.DEFINE_integer('batch_size', 128, 'Batch size of data fed into model')
flags.DEFINE_bool('use_marked_target',
                  True,
                  'Whether to use marked target event(to be predicted) in input data')
flags.DEFINE_bool('use_marked_property',
                  True,
                  'Whether to use marked target property(not to be predicted) in input data')
flags.DEFINE_bool('write_summary', False, 'Whether to write summary of epoch in training using Tensorboard')
flags.DEFINE_float('scale_max', 7, 'Largest scale range')

# Model runner params
flags.DEFINE_integer('max_epoch', 10, 'Max epoch number of training')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate')
flags.DEFINE_float('model_name', 'rmtpp', 'Name for the neural point process')

# Model params
flags.DEFINE_integer('rnn_dim', 32, 'Dimension of LSTM cell')
flags.DEFINE_float('dropout_rate', 0.0, 'Dropout rate of the model')
flags.DEFINE_string('save_dir', 'logs', 'Root path to save logs and models')


def main(argv):
    data_loader = BaseLoader.get_loader_from_flags(FLAGS.data_set)
    train_set, valid_set, test_set = data_loader.load_dataset(FLAGS.batch_size,
                                                              scaled_range_max=FLAGS.scale_max)

    model_runner = ModelRunner(FLAGS)

    model_runner.train(train_set, valid_set, test_set, FLAGS.max_epoch, auto_save=True)
    model_runner.evaluate(test_set)

    return


if __name__ == '__main__':
    app.run(main)
