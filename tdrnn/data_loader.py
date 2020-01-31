import numpy as np
import pickle
from tdrnn.scalers import ZeroMaxScaler
from tdrnn.data_provider import DataProvider


class BaseLoader:
    def __init__(self):
        self.event_num, (self.train_data, self.valid_data, self.test_data) \
            = self.process_datafile()

    def create_data_set(self,
                        data_dict,
                        batch_size,
                        scale_range):
        dtime_scaler = ZeroMaxScaler(scale_range)

        data_provider = DataProvider(event_num=self.event_num,
                                     event_times_seq=data_dict['timestamps'],
                                     event_types_seq=data_dict['types'],
                                     dtime_scaler=dtime_scaler,
                                     batch_size=batch_size)

        return data_provider.gen_data_set()

    def load_dataset(self,
                     batch_size,
                     scaled_range_max=1):
        scale_range = [0, scaled_range_max]

        train_set = self.create_data_set(self.train_data,
                                         batch_size,
                                         scale_range)
        print('Train set shape {}'.format(train_set.shape))

        valid_set = self.create_data_set(self.valid_data,
                                         batch_size,
                                         scale_range)
        print('Valid set shape {}'.format(valid_set.shape))

        test_set = self.create_data_set(self.test_data,
                                        batch_size,
                                        scale_range)
        print('Test set shape {}'.format(test_set.shape))

        return train_set, valid_set, test_set

    def get_event_num(self):
        return self.event_num

    @staticmethod
    def get_loader_from_flags(data_set_name):
        loader_cls = None
        for sub_loader_cls in BaseLoader.__subclasses__():
            if sub_loader_cls.name == data_set_name:
                loader_cls = sub_loader_cls
        if loader_cls is None:
            raise RuntimeError('Unknown dataset:' + data_set_name)

        return loader_cls()

    @staticmethod
    def count_event_num(types_seq):
        min_max_vals = [[min(seq), max(seq)] for seq in types_seq]
        return int(np.max(min_max_vals) - np.min(min_max_vals) + 1)

    @staticmethod
    def split_dataset(dataset, valid_start_ratio=0.8, test_start_ratio=0.9):
        all_datas = [dict(), dict(), dict()]
        num_seqs = len(dataset['types'])
        end_ratios = [valid_start_ratio, test_start_ratio, 1]
        start_idx = 0
        for i, end_ratio in enumerate(end_ratios):
            data_i = all_datas[i]
            end_idx = int(end_ratio * num_seqs)
            for key, value in dataset.items():
                data_i[key] = value[start_idx: end_idx]

            start_idx = end_idx
        return all_datas

    def process_datafile(self):
        examples_ds = None
        event_num = None
        train_data = examples_ds
        valid_data = examples_ds
        test_data = examples_ds

        return event_num, (train_data, valid_data, test_data)

    @staticmethod
    def load_pickle(file_name):
        with open(file_name, 'rb') as f:
            dataset = pickle.load(f, encoding='latin1')
        return dataset


class SoLoader(BaseLoader):
    name = 'data_so'

    def process_datafile(self):
        file = 'data/{}/{}.pkl'

        train_data = self.load_pickle(file.format(self.name, 'train'))
        print('Train dataset:', len(train_data['types']))

        valid_data = self.load_pickle(file.format(self.name, 'valid'))
        print('Valid dataset:', len(valid_data['types']))

        test_data = self.load_pickle(file.format(self.name, 'test'))
        print('Test dataset:', len(test_data['types']))

        event_num = self.count_event_num(test_data['types'])

        return event_num, (train_data, valid_data, test_data)
