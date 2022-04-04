import numpy as np
import torch
import ltn


class TrainingDataLoader:
    """
    Data loader to load the training set of the mindreader dataset.
    """
    def __init__(self,
                 data,
                 batch_size=1,
                 shuffle=True):
        """
        Constructor of the training data loader.

        :param data: list of triples (user, item, rating)
        :param batch_size: batch size for the training of the model
        :param shuffle: whether to shuffle data during training or not
        """
        self.data = np.array(data)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return int(np.ceil(self.data.shape[0] / self.batch_size))

    def __iter__(self):
        n = self.data.shape[0]
        idxlist = list(range(n))
        if self.shuffle:
            np.random.shuffle(idxlist)

        for _, start_idx in enumerate(range(0, n, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, n)
            data = self.data[idxlist[start_idx:end_idx]]
            pos_rat = data[data[:, 2] == 1]
            neg_rat = data[data[:, 2] == -1]

            yield (ltn.Variable('pos_u_idx', torch.tensor(pos_rat[:, 0]), add_batch_dim=False),
                   ltn.Variable('pos_i_idx', torch.tensor(pos_rat[:, 1]), add_batch_dim=False)), \
                  (ltn.Variable('neg_u_idx', torch.tensor(neg_rat[:, 0]), add_batch_dim=False),
                   ltn.Variable('neg_i_idx', torch.tensor(neg_rat[:, 1]), add_batch_dim=False))


class ValDataLoader:
    """
    Data loader to load the validation/test set of the mindreader dataset.
    """
    def __init__(self,
                 data,
                 batch_size=1):
        """
        Constructor of the validation data loader.

        :param data: matrix of user-item pairs. Every row is a user, where the first position contains the positive
        user-item pair, while the last 100 positions contain the negative user-item pairs
        :param batch_size: batch size for the validation/test of the model
        """
        self.data = np.array(data)
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(self.data.shape[0] / self.batch_size))

    def __iter__(self):
        n = self.data.shape[0]
        idxlist = list(range(n))

        for _, start_idx in enumerate(range(0, n, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, n)
            data = self.data[idxlist[start_idx:end_idx]]
            ground_truth = np.zeros((data.shape[0], 101))
            ground_truth[:, 0] = 1

            yield torch.tensor(data), ground_truth