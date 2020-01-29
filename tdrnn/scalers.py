""" A class for scalers """

import numpy as np

class BaseScaler:
    """ Base Scaler """
    def __init__(self, scale_range):
        self.scale_min = scale_range[0]
        self.scale_max = scale_range[1]
        self.scalers = None

    def fit(self, sequences):
        pass

    def scaling(self, sequences):
        return sequences

    def inverse_scaling(self, scaled_sequences):
        return scaled_sequences

    def fit_scaling(self, sequences):
        self.fit(sequences)
        return self.scaling(sequences)


class MinMaxScaler(BaseScaler):
    """
    Scale sequence within sequence min/max range
    The range does not consider the outliers
    """
    def __init__(self, scale_range):
        super(MinMaxScaler, self).__init__(scale_range)

    def fit(self, sequences):
        if self.scalers is not None:
            return

        num_seqs = len(sequences)
        # list of <min_val, max_val>
        self.scalers = np.zeros([num_seqs, 2])

        for i in range(num_seqs):
            seq = np.array(sequences[i])

            min_val, max_val = self.get_regular_min_max_val(seq)

            self.scalers[i] = [min_val, max_val]

    def scaling(self, sequences):
        if self.scalers is None:
            raise RuntimeError('Must fit scaler before inverse transform')
        if len(self.scalers) != len(sequences):
            raise ('The length of the scaled sequences is not the same with the scalers')

        scaled_seqs = [self.scaling_seq(seq, i) for i, seq in enumerate(sequences)]

        return scaled_seqs

    def inverse_scaling(self, scaled_sequences):
        """ Inverse the scaling of all sequences """
        if self.scalers is None:
            raise RuntimeError('Must fit scaler before inverse transform')
        if len(self.scalers) != len(scaled_sequences):
            raise ('The length of the scaled sequences is not the same with the scalers')
        sequences = [self.inverse_scaling_seq(scaled_seq, i) for i, scaled_seq in enumerate(scaled_sequences)]

        return sequences

    def scaling_seq(self, seq, seq_idx):
        min_val = self.scalers[seq_idx][0]
        max_val = self.scalers[seq_idx][1]
        if min_val == max_val:
            return np.array([self.scale_max / 2] * len(seq))
        else:
            return (seq - min_val) / (max_val - min_val) * (self.scale_max - self.scale_min) \
                   + self.scale_min

    def inverse_scaling_seq(self, scaled_seq, seq_idx):
        min_val = self.scalers[seq_idx][0]
        max_val = self.scalers[seq_idx][1]
        if min_val == max_val:
            return np.divide(scaled_seq, (self.scale_max / 2)) * min_val
        else:
            return (scaled_seq - self.scale_min) / (self.scale_max - self.scale_min) \
                   * (max_val - min_val) + min_val

    def get_regular_min_max_val(self, sequence):
        """ Compute min and max without considering outliers """
        std_mul = 3
        mean_val = np.mean(sequence)
        std_val = np.std(sequence)

        mask = (sequence <= mean_val + std_mul * std_val) & (sequence >= mean_val - std_mul * std_val)
        seq_mask = sequence[mask]
        min_val = np.min(seq_mask)
        max_val = np.max(seq_mask)

        return min_val, max_val


class ZeroMaxScaler(MinMaxScaler):
    """
    Scale sequence within sequence 0 - max range
    The range does not consider the outliers
    """
    def __init__(self, scale_range):
        super(ZeroMaxScaler, self).__init__(scale_range)

    def fit(self, seqs):
        """ seqs - 2-dim array or list of list  """

        if self.scalers is not None:
            return

        num_seqs = len(seqs)
        # an array of [min_val, max_val]
        self.scalers = np.zeros([num_seqs, 2])

        for i in range(num_seqs):
            seq = np.array(seqs[i])
            _, max_val = self.get_regular_min_max_val(seq)
            self.scalers[i] = [0.0, max_val]