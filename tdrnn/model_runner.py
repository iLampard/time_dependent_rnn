""" Model Runner that controls the pipeline of running the model   """

from absl import logging
from datetime import datetime as dt
import functools
import numpy as np
import logging as logging_base
import operator
import os
import tensorflow as tf
from tdrnn.models.tf_wrapper import ModelWrapper
from tdrnn.const import RunnerPhase
from tdrnn.metrics import HawkesMetrics


class ModelRunner:
    def __init__(self,
                 model,
                 flags):
        self.model_wrapper = ModelWrapper(model)
        self.model_checkpoint_path = os.path.join(flags.save_dir, flags.data_set)
        self.lr = flags.learning_rate
        self.write_summary = flags.write_summary
        self.rnn_dim = flags.rnn_dim
        self.dropout = flags.dropout_rate
        self.batch_size = flags.batch_size
        self.metric_class = HawkesMetrics()

        logging.get_absl_logger().addHandler(logging_base.StreamHandler())

    @staticmethod
    def num_params():
        """ Return total num of params of the model  """
        total_num = 0
        for var in tf.trainable_variables():
            shape = var.get_shape()
            total_num += functools.reduce(operator.mul, [dim.value for dim in shape], 1)
        return total_num

    def restore(self, restore_path):
        """ Restore model  """
        self.model_wrapper.restore(restore_path)

    def train(self,
              train_dataset,
              valid_dataset,
              test_dataset,
              max_epochs,
              valid_gap_epochs=1,
              auto_save=True):
        """ Run the training """
        folder_path, model_path = self.make_dir()
        logging.get_absl_handler().use_absl_log_file('logs', folder_path)
        logging.info('Start training with max epochs {}'.format(max_epochs))
        logging.info('Model and logs saved in {}'.format(folder_path))
        logging.info('Number of trainable parameters - {}'.format(self.num_params()))
        if self.write_summary:
            self.model_wrapper.init_summary_writer(folder_path)

        best_eval_loss = float('inf')

        for i in range(1, max_epochs + 1):
            loss, _, _ = self.run_one_epoch(train_dataset, RunnerPhase.TRAIN, self.lr)

            # Record the train loss
            loss_metrics = {'loss': np.mean(loss)}

            logging.info('Epoch {0} -- Training loss {1}'.format(i, loss_metrics['loss']))
            self.model_wrapper.write_summary(i, loss_metrics, RunnerPhase.TRAIN)

            # Evaluate the model
            if i % valid_gap_epochs == 0:
                loss, prediction, _ = self.run_one_epoch(valid_dataset, RunnerPhase.VALIDATE, self.lr)
                # Record the validation loss
                loss_metrics = {'loss': np.mean(loss)}
                logging.info('Epoch {0} -- Validation loss {1}'.format(i, loss_metrics['loss']))
                self.model_wrapper.write_summary(i, loss_metrics, RunnerPhase.VALIDATE)

                # If it is the best model on valid set
                if best_eval_loss > loss_metrics['loss']:
                    metrics = self.evaluate(test_dataset)
                    self.model_wrapper.write_summary(i, metrics, RunnerPhase.PREDICT)

                    best_eval_loss = loss_metrics['loss']
                    if auto_save:
                        self.model_wrapper.save(model_path)

        if not auto_save:
            self.model_wrapper.save(model_path)
        else:
            # use the saved model based on valid performance
            self.restore(model_path)

        logging.info('Training finished')

    def evaluate(self, dataset):
        """ Evaluate the model on valid / test set """
        logging.info('Start evaluation')

        loss, predictions, labels = self.run_one_epoch(dataset, RunnerPhase.VALIDATE)

        metrics_dict = self.metric_class.get_metrics_dict(predictions, labels)

        eval_info = self.metric_class.metrics_dict_to_str(metrics_dict)

        logging.info(eval_info)

        logging.info('Evaluation finished')

        return metrics_dict

    def run_one_epoch(self, dataset, phase, lr=None):
        """ Run one complete epoch """
        epoch_loss = []
        epoch_predictions = []
        for x_input in dataset.get_batch_data():
            loss, prediction = self.model_wrapper.run_batch(x_input,
                                                            lr,
                                                            phase=phase)
            epoch_loss.append(loss)
            epoch_predictions.append(prediction)

        epoch_loss = np.array(epoch_loss)

        epoch_predictions = self.concat_element(epoch_predictions)

        if phase == RunnerPhase.PREDICT:
            epoch_predictions = dataset.get_last_inversed_pred(epoch_predictions)
            return epoch_loss, epoch_predictions
        else:
            epoch_predictions, epoch_labels = dataset.get_masked_inversed_pred_and_label(epoch_predictions)
            return epoch_loss, epoch_predictions, epoch_labels

    @staticmethod
    def concat_element(arrs):
        n_len = len(arrs)
        n_element = len(arrs[0])

        concated_outputs = []
        for j in range(n_element):
            a_output = []
            for i in range(n_len):
                a_output.append(arrs[i][j])

            concated_outputs.append(np.concatenate(a_output, axis=0))

        # n_elements * [[n_lens, dim_of_elements]]
        return concated_outputs

    def make_dir(self):
        """ Initialize the directory to save models and logs """
        folder_name = list()
        model_tags = {'lr': self.lr,
                      'dim': self.rnn_dim,
                      'drop': self.dropout}

        for key, value in model_tags.items():
            folder_name.append('{}-{}'.format(key, value))
        folder_name = '_'.join(folder_name)
        current_time = dt.now().strftime('%Y%m%d-%H%M%S')
        folder_path = os.path.join(self.model_checkpoint_path,
                                   self.model_wrapper.__class__.__name__,
                                   folder_name,
                                   current_time)
        os.makedirs(folder_path)
        model_path = os.path.join(folder_path, 'saved_model')
        return folder_path, model_path


class PlateauLRDecay:
    def __init__(self, init_lr, epoch_patience, period_patience, min_lr=0.00001, rate=0.4, verbose=False):
        self.lr = init_lr
        self.epoch_patience = epoch_patience
        self.period_patience = period_patience
        self.min_lr = min_lr
        self.rate = rate
        self.verbose = verbose

        self.prev_best_epoch_num = 0
        self.prev_best_loss = float('inf')

        if self.lr <= self.min_lr:
            self.lr = self.min_lr
            self.is_min_lr = True
        else:
            self.is_min_lr = False

    def update_lr(self, loss, epoch_num):
        if loss < self.prev_best_loss:
            self.prev_best_loss = loss
            self.prev_best_epoch_num = epoch_num
        else:
            epochs = epoch_num - self.prev_best_epoch_num
            if self.is_min_lr is True or epochs >= self.epoch_patience * self.period_patience:
                self.lr = 0.0
            elif epochs % self.epoch_patience == 0:
                # reduce lr
                self.lr = min(self.lr * self.rate, self.min_lr)
                if self.is_min_lr is False and self.lr == self.min_lr:
                    self.is_min_lr = True
                    self.prev_best_epoch_num = epoch_num
                if self.verbose:
                    print('Reduce lr to ', self.lr)
                return True

        return False


class ConstantLR:
    def __init__(self, init_lr):
        self.lr = init_lr

    def update_lr(self, loss, epoch_num):
        return False

    def get_lr(self):
        return self.lr
