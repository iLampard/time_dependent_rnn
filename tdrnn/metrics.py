""" Metrics utilities """

import numpy as np


def flat_seq(seq):
    if isinstance(seq, np.ndarray):
        seq = seq.tolist()

    if isinstance(seq, list):
        res = sum(map(flat_seq, seq), [])
    else:
        res = [seq]

    return res


class HawkesMetrics:
    def get_metrics_dict(self, predictions, labels):
        type_prediction, dtime_prediction = predictions
        type_label, dtime_label = labels

        type_prediction = flat_seq(type_prediction)
        dtime_prediction = flat_seq(dtime_prediction)
        type_label = flat_seq(type_label)
        dtime_label = flat_seq(dtime_label)

        metric_dict = dict()
        metric_dict['type_acc'] = self.accuracy(type_prediction, type_label)
        metric_dict['time_rmse'] = self.rmse(dtime_prediction, dtime_label)
        metric_dict['time_mae'] = self.mae(dtime_prediction, dtime_label)
        metric_dict['time_mape'] = self.mape(dtime_prediction, dtime_label)

        return metric_dict

    @staticmethod
    def rmse(pred, labels):
        """ RMSE ratio """
        return np.sqrt(np.mean(np.subtract(pred, labels) ** 2))

    @staticmethod
    def mae(pred, labels):
        """ MAE ratio """
        return np.mean(np.abs(np.subtract(pred, labels)))

    @staticmethod
    def mape(pred, labels):
        """ MAPE ratio """
        mask = (labels != 0)
        pred = pred[mask]
        labels = labels[mask]

        return np.mean(np.abs(np.subtract(pred, labels)) / labels)

    @staticmethod
    def metrics_dict_to_str(metrics_dict):
        """ Convert metrics to a string to show in the console """
        eval_info = ''
        for key, value in metrics_dict.items():
            eval_info += '{0} : {1}, '.format(key, value)

        return eval_info[:-1]

    @staticmethod
    def loss(total_loss, event_count):
        return np.sum(total_loss) / np.sum(event_count)

    @staticmethod
    def accuracy(pred, label):
        return np.mean(pred == label)
