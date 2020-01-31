""" Dataset for event data """
import numpy as np


class DataSet:
    def __init__(self,
                 x_seqs,
                 batch_size,
                 dtime_scaler,
                 len_seqs,
                 max_len):

        self.x_seqs = x_seqs
        self.batch_size = batch_size

        self.dtime_scaler = dtime_scaler

        self.len_seqs = len_seqs
        self.max_len = max_len

        self.num_x_seqs = len(x_seqs)

    def get_masked_inversed_pred_and_label(self, predictions):
        if len(predictions[0]) != len(self.len_seqs):
            raise RuntimeError('Length of predictions must match that of the input sequences')

        masked_predictions = [[], []]
        masked_labels = [[], []]
        for i in range(len(self.len_seqs)):
            for j in range(self.num_x_seqs):
                masked_predictions[j].append(predictions[j][i][:self.len_seqs[i] - 1])
                masked_labels[j].append(self.x_seqs[j][i][1:self.len_seqs[i]])

        masked_predictions = self.inverse_transform(masked_predictions)
        masked_labels = self.inverse_transform(masked_labels)
        return masked_predictions, masked_labels

    def get_last_inversed_pred(self, predictions):
        if len(predictions[0]) != len(self.len_seqs):
            raise RuntimeError('Length of predictions must match that of the input sequences')

        last_predictions = [[], [], []]
        for i in range(len(self.len_seqs)):
            for j in range(self.num_x_seqs):
                last_predictions[j].append(predictions[j][i][self.len_seqs[i] - 1])

        last_predictions = self.inverse_transform(last_predictions)
        return last_predictions

    def get_batch_data(self):
        x_seqs = self.x_seqs
        num_records = len(x_seqs[0])

        idx = 0
        while idx < num_records:
            next_idx = min(idx + self.batch_size, num_records)
            batch_x = []
            for x_seqs in self.x_seqs:
                batch_x.append(x_seqs[idx: next_idx])

            yield batch_x
            idx = next_idx

    def inverse_transform(self, all_scaled_seqs):
        event_tpye_seq, scaled_dtime_seqs = all_scaled_seqs
        dtime_seqs = self.dtime_scaler.inverse_scaling(scaled_dtime_seqs)

        return event_tpye_seq, dtime_seqs

    @property
    def shape(self):
        return self.x_seqs[0].shape


class DataProvider:
    def __init__(self,
                 event_num,
                 event_times_seq,
                 event_types_seq,
                 dtime_scaler,
                 batch_size):
        self.event_num = event_num
        self.batch_size = batch_size
        self.event_times_seq = event_times_seq
        self.event_types_seq = event_types_seq

        self.dtime_scaler = dtime_scaler

        self.type_padding = event_num
        self.time_padding = 0
        self.marked_target_padding = 0
        self.marked_property_padding = 0

        self.x_seqs, self.len_seqs, self.max_len = self.get_all_seqs()

    def get_all_seqs(self):
        num_seqs = len(self.event_times_seq)

        # dt = t(i+1) - t(i)
        # therefore last event is dropped as no duration of that event can be found
        dtimes_seqs = [[t_seq[i + 1] - t_seq[i] for i in range(len(t_seq) - 1)]
                       for t_seq in self.event_times_seq]

        len_seqs = [len(seq) for seq in dtimes_seqs]

        max_seq_len = max(len_seqs)

        scaled_dtimes_seqs = self.dtime_scaler.fit_scaling(dtimes_seqs)

        x_types_seqs = np.ones([num_seqs, max_seq_len], dtype=np.int32) * self.type_padding
        x_dtimes_seqs = np.ones([num_seqs, max_seq_len], dtype=np.float32) * self.time_padding

        for i in range(num_seqs):
            x_seq_len = len_seqs[i]
            x_types_seqs[i][:x_seq_len] = self.event_types_seq[i][:-1]
            x_dtimes_seqs[i][:x_seq_len] = scaled_dtimes_seqs[i]

        return [x_types_seqs, x_dtimes_seqs], \
               len_seqs, max_seq_len

    def gen_data_set(self):
        return DataSet(self.x_seqs,
                       self.batch_size,
                       self.dtime_scaler,
                       self.len_seqs,
                       self.max_len)
