import tensorflow as tf
import os
from tdrnn.const import RunnerPhase


class ModelWrapper:

    def __init__(self, model):
        """ Init model wrapper """
        self.model = model
        sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        self.sess = tf.Session(config=sess_config)
        self.model.build()

        # initialize saver and tensorboard summary writer
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        self.train_summary_writer = None
        self.valid_summary_writer = None
        self.test_summary_writer = None

    def save(self, checkpoint_path):
        """ Save the model """
        self.saver.save(self.sess, checkpoint_path)

    def restore(self, checkpoint_path):
        """ Restore the model """
        self.saver.restore(self.sess, checkpoint_path)

    def init_summary_writer(self, root_dir):
        """ Init tensorboard writer  """
        tf_board_dir = 'tfb_dir'
        folder = os.path.join(root_dir, tf_board_dir)
        self.train_summary_writer = tf.summary.FileWriter(os.path.join(folder, 'train'), self.sess.graph)
        self.valid_summary_writer = tf.summary.FileWriter(os.path.join(folder, 'valid'))
        self.test_summary_writer = tf.summary.FileWriter(os.path.join(folder, 'test'))

    def write_summary(self, epoch_num, kv_pairs, phase):
        """ Write summary into tensorboard """
        if phase == RunnerPhase.TRAIN:
            summary_writer = self.train_summary_writer
        elif phase == RunnerPhase.VALIDATE:
            summary_writer = self.valid_summary_writer
        elif phase == RunnerPhase.PREDICT:
            summary_writer = self.test_summary_writer
        else:
            raise RuntimeError('Unknow phase: ' + phase)

        if summary_writer is None:
            return

        for key, value in kv_pairs.items():
            metrics = tf.Summary()
            metrics.value.add(tag=key, simple_value=value)
            summary_writer.add_summary(metrics, epoch_num)

        summary_writer.flush()

    def run_batch(self, batch_data, lr, phase):
        """ Run one batch """
        if phase == RunnerPhase.TRAIN:
            loss, prediction = self.model.train(self.sess, batch_data, lr=lr)
        else:
            loss, prediction = self.model.predict(self.sess, batch_data)

        return loss, prediction
