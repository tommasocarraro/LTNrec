import copy
import json
import pandas as pd
import os
import numpy as np


class MindReaderDataset:
    """
    Class that manages the MindReader dataset.
    """
    def __init__(self, data_path):
        self.check(data_path)
        self.uri_to_idx, self.idx_to_uri = self.get_mr_uri_idx_mapping(data_path)
        self.genre_to_idx = self.get_genre_to_idx(data_path)
        self.year_to_idx = self.get_year_to_idx(data_path)
        self.movies_uri, self.genres_uri, self.decades_uri = self.get_movie_genres_decades_uri(data_path)
        self.movie_to_idx = self.get_movie_to_idx()
        self.split = self.read_split(data_path)
        self.uri_to_genre = self.get_uri_to_genre(data_path)
        self.validation = self.get_validation()
        self.test = self.get_test()
        self.movie_to_genres = self.get_movie_genres_map(data_path)
        self.movie_to_year = self.get_movie_year_map(data_path)
        self.user_movie_ratings = self.get_user_movie_ratings()
        self.user_genre_ratings = self.get_user_genre_ratings()
        self.n_movies = len(self.movies_uri)
        self.n_genres = len(self.genres_uri)
        self.n_users = len(self.split['training'])

    @staticmethod
    def check(path):
        """
        Check the presence of all needed files:
        1. entities: contains the entities (movies, genres, actors, ...) of the MindReader KG;
        2. meta: contains the mapping between the URIs of the entities and unique identifiers;
        3. split: contains a split of the MindReader dataset, subdivided into training, validation, and testing data;
        4. triples: contains the relationships between the entities of the MindReader KG. For example, who is the actor
        of the movie? Which is the genre? Which is the year? All this information is included in the triples file.
        """
        assert os.path.exists(os.path.join(path, 'entities.csv')), "entities.csv is missing"
        assert os.path.exists(os.path.join(path, 'meta.json')), "meta.json is missing"
        assert os.path.exists(os.path.join(path, 'split.json')), "split.json is missing"
        assert os.path.exists(os.path.join(path, 'triples.csv')), "triples.csv is missing"

    @staticmethod
    def get_mr_uri_idx_mapping(path):
        """
        Returns the mapping of the MindReader dataset. In the knowledge graph, the entities are identified by a URI.
        In the MindReader dataset there is a mapping between these URIs and integer unique identifiers.
        This function returns two mappings. One from URIs to ids, and one from ids to URIs.
        """
        with open(os.path.join(path, 'meta.json')) as f:
            uri_to_idx = json.load(f)['e_idx_map']
            idx_to_uri = {v: k for k, v in uri_to_idx.items()}
            return uri_to_idx, idx_to_uri

    def get_genre_to_idx(self, path):
        """
        Returns a mapping between the labels of the genres in the MindReader dataset and a unique identifier for the
        genres.
        """
        genre_to_idx = {}
        entities = pd.read_csv(os.path.join(path, "entities.csv"))
        genres_rows = entities.loc[['Genre' in labels.split('|')
                                    for labels in list(entities['labels'])]]
        c = 0
        for g in list(genres_rows['name']):
            if g not in genre_to_idx:
                genre_to_idx[g] = c
                c += 1
        return genre_to_idx

    def get_year_to_idx(self, path):
        """
        Returns a mapping between the labels of the decades in the MindReader dataset and a unique identifier for the
        years.
        """
        year_to_idx = {}
        entities = pd.read_csv(os.path.join(path, "entities.csv"))
        decades_rows = entities.loc[['Decade' in labels for labels in list(entities['labels'])]]
        c = 0
        for y in list(decades_rows['uri']):
            if y not in year_to_idx:
                year_to_idx[y] = c
                c += 1
        return year_to_idx

    @staticmethod
    def read_split(path):
        """
        Reads the split file and returns a dictionary containing the split data (training, validation, test data).
        """
        with open(os.path.join(path, "split.json")) as f:
            return json.load(f)

    def get_validation(self):
        """
        Returns the validation data contained in the split of the dataset. The validation data is organized as follows.
        For each validation user, we have one positive interaction and 100 randomly sampled negative interactions.
        So, for each user, we have 101 user-item pairs, where the last pair represents the positive interaction
        while the first 100 pairs represent the negative interactions.
        """
        return [np.concatenate(([[u, self.movie_to_idx[self.idx_to_uri[n]]] for n in negs],
                                [[u, self.movie_to_idx[self.idx_to_uri[pos]]]]), axis=0)
                for u, (pos, negs) in self.split['validation']]

    def get_test(self):
        """
        Returns the test data contained in the split of the dataset. The test data is organized as follows.
        For each validation user, we have one positive interaction and 100 randomly sampled negative interactions.
        So, for each user, we have 101 user-item pairs, where the last pair represents the positive interaction
        while the first 100 pairs represent the negative interactions.
        """
        return [np.concatenate(([[u, self.movie_to_idx[self.idx_to_uri[n]]] for n in negs],
                               [[u, self.movie_to_idx[self.idx_to_uri[pos]]]]), axis=0)
                for u, (pos, negs) in self.split['testing']]

    def get_movie_genres_decades_uri(self, path):
        """
        Returns a list containing the URIs of the movies in the MindReader dataset, a list containing the URIs
        of the genres in the MindReader dataset, and a list containing the URIs of the decades in the MindReader
        dataset.
        """
        entities = pd.read_csv(os.path.join(path, "entities.csv"))
        genres_rows = entities.loc[['Genre' in labels.split('|')
                                    for labels in list(entities['labels'])]]
        year_rows = entities.loc[['Decade' in labels for labels in list(entities['labels'])]]
        return list(entities.loc[['Movie' in labels.split('|') and uri in self.uri_to_idx
                                  for uri, labels in zip(list(entities['uri']), list(entities['labels']))]]['uri']), \
               list(genres_rows['uri']), list(year_rows['uri'])

    def get_movie_genres_map(self, path):
        """
        Returns the mapping between movies and their genres. The movies and the genres are represented with their
        unique identifiers. Every movie is associated with a list of indexes of genres.
        """
        triples = pd.read_csv(os.path.join(path, "triples.csv"))
        genre_triples = triples.loc[triples['relation'] == 'HAS_GENRE']

        movie_genre_map = {}
        for m, g in zip(list(genre_triples['head_uri']), list(genre_triples['tail_uri'])):
            # since the relationships are undirected, we need only one direction. Here, we select the direction
            # from movie to genre, and not viceversa, since the information is redundant
            # then, we need to check if the movie involved in the relationship has been included in the
            # dataset. For this reason, we check if its id is in the meta.json file. In fact, not all the movies are
            # included in the dataset.
            if m in self.movies_uri:
                if self.movie_to_idx[m] not in movie_genre_map:
                    movie_genre_map[self.movie_to_idx[m]] = [self.genre_to_idx[self.uri_to_genre[g]]]
                else:
                    movie_genre_map[self.movie_to_idx[m]].append(self.genre_to_idx[self.uri_to_genre[g]])

        return movie_genre_map

    def get_movie_year_map(self, path):
        """
        Returns the mapping between movies and their decades. The movies and the decades are represented with their
        unique identifiers. Every movie is associated with an index of decade.
        """
        triples = pd.read_csv(os.path.join(path, "triples.csv"))
        decade_triples = triples.loc[triples['relation'] == 'FROM_DECADE']

        movie_decade_map = {}
        for m, y in zip(list(decade_triples['head_uri']), list(decade_triples['tail_uri'])):
            # since the relationships are undirected, we need only one direction. Here, we select the direction
            # from movie to genre, and not viceversa, since the information is redundant
            # then, we need to check if the movie involved in the relationship has been included in the
            # dataset. For this reason, we check if its id is in the meta.json file. In fact, not all the movies are
            # included in the dataset.
            if m in self.movies_uri:
                movie_decade_map[self.movie_to_idx[m]] = self.year_to_idx[y]

        return movie_decade_map

    @staticmethod
    def get_uri_to_genre(path):
        """
        Returns a mapping between the URIs of the movies' genres in the MindReader dataset and their name (e.g., Drama
        Film).
        """
        entities = pd.read_csv(os.path.join(path, "entities.csv"))
        genres_rows = entities.loc[['Genre' in labels.split('|') for labels in list(entities['labels'])]]
        uri_to_genre = {}
        for uri, g in zip(list(genres_rows['uri']), list(genres_rows['name'])):
            uri_to_genre[uri] = g

        return uri_to_genre

    def get_movie_to_idx(self):
        """
        Create a mapping between URIs of movies and unique identifiers starting from zero. This is used because
        MindReader's unique identifiers are mixed between movies and genres. Here, we need two different mappings.
        """
        c = 0
        movie_to_idx = {}
        for m in self.movies_uri:
            movie_to_idx[m] = c
            c += 1
        return movie_to_idx

    def get_user_movie_ratings(self, keep=None, seed=None):
        """
        Returns the ratings of the users for the movies of the MindReader dataset. The ratings are given in the form
        of triples:
        1. user_idx;
        2. movie_idx;
        3. rating: 1 or -1.

        :param keep: percentage of ratings to be kept for each user. If None or False, all the ratings of the dataset
        are kept. If True, the given percentage of ratings is randomly drawn from the ratings of each user.
        :param seed: seed used to randomly pick training ratings for each user in the case `keep` is not None
        """
        filtered_train = copy.deepcopy(self.split["training"])
        if keep is not None:
            if seed:
                np.random.seed(seed)
            for i, (u, ratings) in enumerate(filtered_train):
                n_keep = np.ceil(len(ratings) * keep)
                np.random.shuffle(ratings)
                filtered_train[i][1] = ratings[:int(n_keep)]
        return [(u, self.movie_to_idx[self.idx_to_uri[r['e_idx']]], r['rating'])
                for u, ratings in filtered_train for r in ratings if len(r) > 0 and r['is_movie_rating']]

    def get_user_genre_ratings(self):
        """
        Returns the ratings of the users for the genres of the MindReader dataset. The ratings are given in the form
        of triples:
        1. user_idx;
        2. genre_idx;
        3. rating: 1 or -1.
        """
        return [(u, self.genre_to_idx[self.uri_to_genre[self.idx_to_uri[r['e_idx']]]], r['rating'])
                for u, ratings in self.split['training']
                for r in ratings if self.idx_to_uri[r['e_idx']] in self.genres_uri]

    def get_user_decades_ratings(self):
        """
        Returns the ratings of the users for the decades of the MindReader dataset. The ratings are given in the form
        of triples:
        1. user_idx;
        2. decade_idx;
        3. rating: 1 or -1.
        """
        return [(u, self.year_to_idx[self.idx_to_uri[r['e_idx']]], r['rating'])
                for u, ratings in self.split['training']
                for r in ratings if self.idx_to_uri[r['e_idx']] in self.decades_uri]

    def get_user_genre_ratings_dict(self):
        """
        Returns the ratings of the users for the genres of the MindReader dataset. It returns a dictionary with the
        following structure. For each user, there are two dictionaries. The first one contains the genres that the user
        likes, while the second one contains the genres that the user dislikes.
        """
        user_genres = {}
        for u, ratings in self.split['training']:
            user_genres[u] = {'likes': [], 'dislikes': []}
            for r in ratings:
                if self.idx_to_uri[r['e_idx']] in self.genres_uri:
                    if r['rating'] == 1:
                        user_genres[u]['likes'].append(
                            self.genre_to_idx[self.uri_to_genre[self.idx_to_uri[r['e_idx']]]])
                    else:
                        user_genres[u]['dislikes'].append(
                            self.genre_to_idx[self.uri_to_genre[self.idx_to_uri[r['e_idx']]]])
        return user_genres

    def get_user_decade_ratings_dict(self):
        """
        Returns the ratings of the users for the decades of the MindReader dataset. It returns a dictionary with the
        following structure. For each user, there are two dictionaries. The first one contains the decades that the user
        likes, while the second one contains the decades that the user dislikes.
        """
        user_decades = {}
        for u, ratings in self.split['training']:
            user_decades[u] = {'likes': [], 'dislikes': []}
            for r in ratings:
                if self.idx_to_uri[r['e_idx']] in self.decades_uri:
                    if r['rating'] == 1:
                        user_decades[u]['likes'].append(
                            self.year_to_idx[self.idx_to_uri[r['e_idx']]])
                    else:
                        user_decades[u]['dislikes'].append(
                            self.year_to_idx[self.idx_to_uri[r['e_idx']]])
        return user_decades

    def get_user_items_genre(self, dislikes=True, include_tr_ratings=True):
        """
        Returns all the user-item pairs (u, i) of the MindReader dataset for which the user u dislikes/likes at least
        one of the genres of the item i.

        :param dislikes: if True, the function returns the u-i pairs for which the user dislikes at least one genre
        of i. If False, the function returns the u-i pairs for which the user likes at least of genre of i.
        :param include_tr_ratings: whether the returned user-item pairs have to include also the ratings for the movies
        in the training set. Default to True.
        """
        u_i_genre = []
        user_genre_ratings = self.get_user_genre_ratings_dict()
        for u in range(self.n_users):
            for i in range(self.n_movies):
                m_genres = set(self.movie_to_genres[i])  # genres of item i
                u_genres = set(user_genre_ratings[u]['dislikes']) if dislikes else set(user_genre_ratings[u]['likes'])  # genres that user u does not like/like
                if m_genres.intersection(u_genres):  # there is at least on genre in i that the user u does not like
                    u_i_genre.append((u, i))

        if not include_tr_ratings:
            u_i_genre = set(u_i_genre) - set([(u, i) for u, i, r in self.get_user_movie_ratings()])
            u_i_genre = list(u_i_genre)

        return np.array(u_i_genre)

    def get_t_most_popular_genres(self, t):
        """
        Returns the indexes of the t most popular genres of the MindReader dataset. The popularity is given by the
        number of ratings that the genre has in the dataset. Higher the number, more popular the genre is.

        :param t: threshold for taking the most popular genres from the dataset
        :return: the indexes of the t most popular genres
        """
        idx_to_genre = {v: k for k, v in self.genre_to_idx.items()}
        ratings_per_genre = np.zeros(self.n_genres)
        for u, g, r in self.get_user_genre_ratings():
            ratings_per_genre[g] += 1

        sorted_genres = np.argsort(-ratings_per_genre)
        print({idx_to_genre[g]: ratings_per_genre[g] for g in sorted_genres[:t]})
        return sorted_genres[:t]
