import numpy as np
import torch
import ltn


class TrainingDataLoaderLTN:
    """
    Data loader to load the training set of the MindReader dataset. It creates batches and wrap them inside LTN
    variables ready for the learning.
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
            ratings = data[:, -1]
            ratings[ratings == -1] = 0

            yield ltn.Variable('users', torch.tensor(data[:, 0]), add_batch_dim=False),\
                  ltn.Variable('items', torch.tensor(data[:, 1]), add_batch_dim=False),\
                  ltn.Variable('ratings', torch.tensor(ratings), add_batch_dim=False)


class TrainingDataLoader:
    """
    Data loader to load the training set of the MindReader dataset. It creates batches composed of user-item pairs
    and their corresponding ratings.
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
            u_i_pairs = data[:, :2]
            ratings = data[:, -1]
            ratings[ratings == -1] = 0

            yield torch.tensor(u_i_pairs), torch.tensor(ratings).float()


class ValDataLoader:
    """
    Data loader to load the validation/test set of the MindReader dataset.
    """
    def __init__(self,
                 data,
                 batch_size=1):
        """
        Constructor of the validation data loader.

        :param data: matrix of user-item pairs. Every row is a user, where the last position contains the positive
        user-item pair, while the first 100 positions contain the negative user-item pairs
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
            ground_truth[:, -1] = 1

            yield torch.tensor(data), ground_truth
