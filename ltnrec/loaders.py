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

            yield ltn.Variable('users', torch.tensor(data[:, 0]), add_batch_dim=False), \
                  ltn.Variable('items', torch.tensor(data[:, 1]), add_batch_dim=False), \
                  ltn.Variable('ratings', torch.tensor(ratings), add_batch_dim=False)


class TrainingDataLoaderLTNGenres:
    """
    Data loader to load the training set of the MindReader dataset. It creates batches and wrap them inside LTN
    variables ready for the learning. This loaders differs from the TrainingDataLoaderLTN. In particular, it creates
    LTN variables to reason on formulas which involve the genres of the movies.
    """

    def __init__(self,
                 movie_ratings,
                 genre_ratings,
                 movie_genre_map,
                 batch_size=1,
                 shuffle=True):
        """
        Constructor of the training data loader.

        :param movie_ratings: list of triples (user, item, rating)
        :param genre_ratings: dictionary containing the genres that each user likes and dislikes
        :param movie_genre_map: dictionary containing a list of genres for each movie
        :param batch_size: batch size for the training of the model
        :param shuffle: whether to shuffle data during training or not
        """
        self.movie_ratings = np.array(movie_ratings)
        self.genre_ratings = genre_ratings
        self.movie_genre_map = movie_genre_map
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return int(np.ceil(self.movie_ratings.shape[0] / self.batch_size))

    def __iter__(self):
        n = self.movie_ratings.shape[0]
        idxlist = list(range(n))
        if self.shuffle:
            np.random.shuffle(idxlist)

        for _, start_idx in enumerate(range(0, n, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, n)
            data = self.movie_ratings[idxlist[start_idx:end_idx]]

            users = data[:, 0]
            items = data[:, 1]
            ratings = data[:, 2]
            ratings[ratings == -1] = 0

            # for each item in the batch, get the set of its genres
            movie_genres_map = [(i, set(self.movie_genre_map[i])) for i in set(items)]
            # for each user in the the batch, get the genres he/she dislikes
            user_genres_map_dislike = [(u, set(self.genre_ratings[u]['dislikes'])) for u in set(users)]
            # user_genres_map_like = [set(self.genre_ratings[u]['likes']) for u in users]

            # now, I create all the user-item pairs in the batch for which I can decrease the Likes predicate based
            # on the axiom not(LikesGenre(u,g)) and HasGenre(i,g) -> not(Likes(u,i))
            u_i_decrease = np.array([(u, m) for m, m_genres in movie_genres_map
                                     for u, u_genres in user_genres_map_dislike for g in u_genres if g in m_genres])

            yield (ltn.Variable('users', torch.tensor(data[:, 0]), add_batch_dim=False),
                   ltn.Variable('items', torch.tensor(data[:, 1]), add_batch_dim=False),
                   ltn.Variable('ratings', torch.tensor(ratings), add_batch_dim=False)), \
                  (ltn.Variable('u_decrease', torch.tensor(u_i_decrease[:, 0]), add_batch_dim=False),
                   ltn.Variable('i_decrease', torch.tensor(u_i_decrease[:, 1]), add_batch_dim=False),
                   ltn.Variable('gt', torch.zeros(u_i_decrease.shape[0]), add_batch_dim=False)) \
                      if u_i_decrease.size else (None, None, None)


class TrainingDataLoaderLTNDecades:
    """
    Data loader to load the training set of the MindReader dataset. It creates batches and wrap them inside LTN
    variables ready for the learning. This loaders differs from the TrainingDataLoaderLTN. In particular, it creates
    LTN variables to reason on formulas which involve the genres of the movies.
    """

    def __init__(self,
                 movie_ratings,
                 decade_ratings,
                 movie_decade_map,
                 batch_size=1,
                 shuffle=True):
        """
        Constructor of the training data loader.

        :param movie_ratings: list of triples (user, item, rating)
        :param genre_ratings: dictionary containing the genres that each user likes and dislikes
        :param movie_genre_map: dictionary containing a list of genres for each movie
        :param batch_size: batch size for the training of the model
        :param shuffle: whether to shuffle data during training or not
        """
        self.movie_ratings = np.array(movie_ratings)
        self.decade_ratings = decade_ratings
        self.movie_decade_map = movie_decade_map
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return int(np.ceil(self.movie_ratings.shape[0] / self.batch_size))

    def __iter__(self):
        n = self.movie_ratings.shape[0]
        idxlist = list(range(n))
        if self.shuffle:
            np.random.shuffle(idxlist)

        for _, start_idx in enumerate(range(0, n, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, n)
            data = self.movie_ratings[idxlist[start_idx:end_idx]]

            users = data[:, 0]
            items = data[:, 1]
            ratings = data[:, 2]
            ratings[ratings == -1] = 0

            # for each item in the batch, get the set of its genres
            movie_decades_map = [(i, {self.movie_decade_map[i]}) for i in set(items)]
            # for each user in the the batch, get the genres he/she dislikes
            user_decades_map_dislike = [(u, set(self.decade_ratings[u]['dislikes'])) for u in set(users)]
            # user_genres_map_like = [set(self.genre_ratings[u]['likes']) for u in users]

            # now, I create all the user-item pairs in the batch for which I can decrease the Likes predicate based
            # on the axiom not(LikesGenre(u,g)) and HasGenre(i,g) -> not(Likes(u,i))
            u_i_decrease = np.array([(u, m) for m, m_decades in movie_decades_map
                                     for u, u_decades in user_decades_map_dislike for g in u_decades if g in m_decades])

            yield (ltn.Variable('users', torch.tensor(data[:, 0]), add_batch_dim=False),
                   ltn.Variable('items', torch.tensor(data[:, 1]), add_batch_dim=False),
                   ltn.Variable('ratings', torch.tensor(ratings), add_batch_dim=False)), \
                  (ltn.Variable('u_decrease', torch.tensor(u_i_decrease[:, 0]), add_batch_dim=False),
                   ltn.Variable('i_decrease', torch.tensor(u_i_decrease[:, 1]), add_batch_dim=False),
                   ltn.Variable('gt', torch.zeros(u_i_decrease.shape[0]), add_batch_dim=False)) \
                      if u_i_decrease.size else (None, None, None)


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
