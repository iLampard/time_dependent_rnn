import tensorflow as tf
from tensorflow.keras import layers


class TDRNN:
    def __init__(self,
                 process_dim,
                 hidden_dim,
                 **kwargs):
        self.process_dim = process_dim
        self.hidden_dim = hidden_dim

    def build(self):
        with tf.variable_scope('TDRNN'):
            self.types_seq = tf.placeholder(tf.int32, shape=[None, None])
            self.dtimes_seq = tf.placeholder(tf.float32, shape=[None, None])
            self.len_seq = tf.placeholder(tf.float32, shape=[None, 1])