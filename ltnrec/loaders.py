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
            movie_genres_map = [(i, set(self.movie_genre_map[i])) for i in items]
            # for each user in the the batch, get the genres he/she dislikes
            user_genres_map_dislike = [(u, set(self.genre_ratings[u]['dislikes'])) for u in users]
            # user_genres_map_like = [set(self.genre_ratings[u]['likes']) for u in users]

            # now, I create all the user-item pairs in the batch for which I can decrease the Likes predicate based
            # on the axiom not(LikesGenre(u,g)) and HasGenre(i,g) -> not(Likes(u,i))
            u_i_decrease = np.array([(u, m) for m, m_genres in movie_genres_map
                                     for u, u_genres in user_genres_map_dislike for g in u_genres if g in m_genres])

            # todo forse sarebbe meglio non usare solo i rating del batch, ossia non usare il diag, quindi dovrei prendere
            # tutti i likes genre dello user, poi tutti gli item del batch per i quali ho questi genre, e da li creare le triple
            # che mi servono per il not likes. In questo modo non uso solo (si cazzarola, uso solo quello, che palle!)
            # vorrei usare tutti gli item che hanno quel genere, ma non e' possibile fare a batch in quel modo
            # in realta', nei varu batch ci sono potenzialmente tutte le possibilita', perche' se ho un item nel dataset
            # allora lo ho anche sul training, almeno su un utente
            # tutti gli utenti di training sono anche in validation e test
            # todo fare esperimento della rimozione di parte dei dati

            # per ogni coppia user-item del batch, vedo se all'utente u non piace almeno un genere di i, e in quel caso
            # diminuisco il likes per quella coppia
            # qui faccio l'intersezione tra i generi che a u non piacciono e i generi di i
            #genres_for_rule = [list(i_g.intersection(u_g)) for (i, i_g), (u, u_g) in zip(movie_genres_map, user_genres_map_dislike)]

            # se almeno un genere e' nell'intersezione, allora diminuisco quella coppia u-i
            #u_i_triples = np.array([(u, i) for u, i, g in zip(users, items, genres_for_rule) if g])

            yield (ltn.Variable('users', torch.tensor(data[:, 0]), add_batch_dim=False),
                   ltn.Variable('items', torch.tensor(data[:, 1]), add_batch_dim=False),
                   ltn.Variable('ratings', torch.tensor(ratings), add_batch_dim=False)), \
                  (ltn.Variable('u', torch.tensor(u_i_decrease[:, 0]), add_batch_dim=False),
                   ltn.Variable('i', torch.tensor(u_i_decrease[:, 1]), add_batch_dim=False))
                   #ltn.Variable('g', torch.tensor(u_i_g_triples[:, 2]), add_batch_dim=False))


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
