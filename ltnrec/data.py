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
        self.movies_uri, self.genres_uri = self.get_movie_genres_uri(data_path)
        self.movie_to_idx = self.get_movie_to_idx()
        self.split = self.read_split(data_path)
        self.uri_to_genre = self.get_uri_to_genre(data_path)
        self.validation = self.get_validation()
        self.test = self.get_test()
        self.movie_to_genres = self.get_movie_genres_map(data_path)
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
        genres. These unique identifiers have to be used as indexes for a multi-hot vector containing the genres of
        one specific movie. For example, a vector could be 1 0 ... 0 0 1, meaning that the genres at position one and
        on the last position are genres of the current movie.
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
        So, for each user, we have 101 user-item pairs, where the first pair represents the positive interaction
        while the last 100 pairs represent the negative interactions.
        """
        return [np.concatenate(([[u, self.movie_to_idx[self.idx_to_uri[pos]]]],
                                [[u, self.movie_to_idx[self.idx_to_uri[n]]] for n in negs]), axis=0)
                for u, (pos, negs) in self.split['validation']]

    def get_test(self):
        """
        Returns the test data contained in the split of the dataset. The test data is organized as follows.
        For each validation user, we have one positive interaction and 100 randomly sampled negative interactions.
        So, for each user, we have 101 user-item pairs, where the first pair represents the positive interaction
        while the last 100 pairs represent the negative interactions.
        """
        return [np.concatenate(([[u, self.movie_to_idx[self.idx_to_uri[pos]]]],
                                [[u, self.movie_to_idx[self.idx_to_uri[n]]] for n in negs]), axis=0)
                for u, (pos, negs) in self.split['testing']]

    def get_movie_genres_uri(self, path):
        """
        Returns a list containing the URIs of the movies in the MindReader dataset, and a list containing the URIs
        of the genres in the MindReader dataset.
        """
        entities = pd.read_csv(os.path.join(path, "entities.csv"))
        genres_rows = entities.loc[['Genre' in labels.split('|')
                                    for labels in list(entities['labels'])]]
        return list(entities.loc[['Movie' in labels.split('|') and uri in self.uri_to_idx
                                  for uri, labels in zip(list(entities['uri']), list(entities['labels']))]]['uri']), \
               list(genres_rows['uri'])

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

    def get_user_movie_ratings(self):
        """
        Returns the ratings of the users for the movies of the MindReader dataset. The ratings are given in the form
        of triples:
        1. user_idx;
        2. movie_idx;
        3. rating: 1 or -1.
        """
        return [(u, self.movie_to_idx[self.idx_to_uri[r['e_idx']]], r['rating'])
                for u, ratings in self.split['training'] for r in ratings if r['is_movie_rating']]

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