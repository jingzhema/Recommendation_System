import pandas as pd
import numpy as np
from ast import literal_eval
from datetime import datetime
from surprise import Reader, Dataset, SVD
from surprise.model_selection import KFold, cross_validate
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import combinations
from scipy.sparse import csr_matrix
# from surprise.accuracy import rmse
from collections import defaultdict
# from surprise import accuracy
from sklearn.decomposition import TruncatedSVD

class Recommendater:
    def __init__(self, user_id=0, method=''):
        self.user_id = user_id
        self.method = method

        self.USER_COL = 'userId'
        self.ITEM_COL = 'movieId'
        self.RATING_COL = 'rating'

        self.links = pd.read_csv('./data/links.csv', low_memory=False)
        self.links_small = pd.read_csv('./data/links_small.csv', low_memory=False)
        self.meta_data = pd.read_csv('./data/movies_metadata.csv', low_memory=False)
        self.ratings_small = pd.read_csv('./data/ratings_small.csv', low_memory=False)
        self.meta_data['genres'] = self.meta_data['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else []).apply(lambda x: ','.join([str(i) for i in x]))
        self.movies = self.meta_data[['id', 'title', 'genres', 'vote_average', 'homepage']]

    def genres_based(self, movie_title):

        # Map the most frequent words to feature indices and hence compute a word occurrence frequency (sparse) matrix.
        tf = TfidfVectorizer(analyzer=lambda s: (c for i in range(1,4) for c in combinations(s.split(','), r=i)))
        tfidf_matrix = tf.fit_transform(self.movies['genres'])

        # compute similarity matrix
        cosine_sim = cosine_similarity(tfidf_matrix)

        movies = self.movies.reset_index()
        titles = movies['title']
        indices = pd.Series(movies.index, index=movies['title'])
        if movie_title in titles.values:
            idx = indices[movie_title]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:31]
            movie_indices = [i[0] for i in sim_scores]
            recommendations = titles.iloc[movie_indices].head(10)
            data = movies.merge(recommendations.to_frame(), left_index=True, right_index=True)
            data.rename(columns={'title_x': 'Movie Title', 'genres': 'Genres', 'vote_average': 'Vote Average'}, inplace=True)
            data.reset_index(drop=True, inplace=True)
            return data[['Movie Title', 'Genres', 'Vote Average']]
        else:
            return pd.DataFrame()

    def get_top_n(self, predictions, n=10):
        """Return the top-N recommendation for each user from a set of predictions.
        Args:
            predictions(list of Prediction objects): The list of predictions, as
                returned by the test method of an algorithm.
            n(int): The number of recommendation to output for each user. Default
                is 10.
        Returns:
        A dict where keys are user (raw) ids and values are lists of tuples:
            [(raw item id, rating estimation), ...] of size n.
        """

        # First map the predictions to each user.
        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            top_n[uid].append((iid, est))

        # Then sort the predictions for each user and retrieve the k highest ones.
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = user_ratings[:n]

        return top_n

    # Memory-based methods use user rating historical data to compute the similarity between users or items.
    # Collaborative Filtering is based on the idea that users similar to a me can be used to predict how much I will like a particular movie.
    # Singular Value Decomposition (SVD) to minimise RMSE (Root Mean Square Error) and give great recommendations.
    def memory_based(self, user_id='1'):
        reader = Reader()
        data = Dataset.load_from_df(self.ratings_small[['userId', 'movieId', 'rating']], reader)
        svd = SVD()
        trainset = data.build_full_trainset()
        svd.fit(trainset)
        testset = trainset.build_anti_testset()
        predictions = svd.test(testset)
        top_n = self.get_top_n(predictions, 10)

        result = ""
        df_recommendation = pd.DataFrame(columns=['id'])
        if (user_id.isnumeric() and int(user_id) > 0):
            for item in top_n.get(int(user_id)):
                df_recommendation = df_recommendation.append({'id': item[0]}, ignore_index=True)

            result = df_recommendation.merge(self.movies, on='id', how='left')
            result.rename(columns={'title': 'Movie Title', 'genres': 'Genres', 'vote_average': 'Vote Average'}, inplace=True)

        return result[['Movie Title', 'Genres', 'Vote Average']]

    def top_k_items_for_user(user_id, top_k, corr_mat):
        top_items = corr_mat[:,user_id-1].argsort()[-top_k:][::-1]
        return top_items


    def memory_based2(self, user_id):
        # preprocess data
        row = self.ratings_small[self.USER_COL]
        col = self.ratings_small[self.ITEM_COL]
        data = self.ratings_small[self.RATING_COL]

        # init user-item matrix
        mat = csr_matrix((data, (row, col)))
        mat.eliminate_zeros()

        item_corr_mat = cosine_similarity(mat)

        # similar_items = self.top_k_items_for_user(1, top_k = 10, corr_mat = item_corr_mat)

        similar_items = item_corr_mat[:, int(user_id)].argsort()[-10:][::-1]

        return pd.DataFrame()



    # def truncated_svd(self, user_id):
    #     epsilon = 1e-9
    #     n_latent_factors = 10
    #
    #     row = self.ratings_small['userId']
    #     col = self.ratings_small['movieId']
    #     data = self.ratings_small['rating']
    #
    #     # preprocessing
    #     rated_items = items.loc[items[ITEM_COL].isin(ratings[ITEM_COL])].copy()
    #
    #     mat = csr_matrix((data, (row, col)), shape=(self.NUM_USERS, self.NUM_ITEMS))
    #     mat.eliminate_zeros()
    #
    #     # calculate item latent matrix
    #     item_svd = TruncatedSVD(n_components=n_latent_factors)
    #     item_features = item_svd.fit_transform(mat.transpose()) + epsilon
    #
    #     # calculate user latent matrix
    #     user_svd = TruncatedSVD(n_components=n_latent_factors)
    #     user_features = user_svd.fit_transform(mat) + epsilon
    #
    #     # compute similarity
    #     item_corr_mat = cosine_similarity(item_features)
    #
    #     # get top k item
    #     similar_items = self.top_k_items(name2ind['99'],
    #                                 top_k = 10,
    #                                 corr_mat = item_corr_mat,
    #                                 map_name = ind2name)
    #
    #     return self.ratings_small.loc[self.ratings_small['movieId'].isin(similar_items)


    def top_k_items(self, item_id, top_k, corr_mat, map_name):
        # sort correlation value ascendingly and select top_k item_id
        top_items = corr_mat[item_id,:].argsort()[-top_k:][::-1]
        top_items = [map_name[e] for e in top_items]

        return top_items

    def get_user_ratings(self, user_id):
        data = self.ratings_small[self.ratings_small['userId']==int(user_id)]
        result = pd.merge(left=data, right=self.movies, left_on='movieId', right_on='id')
        result.rename(columns={'userId': 'User ID', 'movieId': 'Movie ID', 'rating': 'Rating', 'title': 'Movie Title', 'genres': 'Genres'}, inplace=True)

        return result[['User ID', 'Movie ID', 'Rating', 'Movie Title', 'Genres']]


    def content_based(self, movie_title):
        # rated_items = self.movies.loc[self.movies['id'].isin(self.ratings_small[self.ITEM_COL])].copy()
        rated_items = self.movies

        # extract the genre
        genre = rated_items['genres'].str.split(",", expand=True)

        # get all possible genre
        all_genre = set()
        for c in genre.columns:
            distinct_genre = genre[c].str.lower().str.strip().unique()
            all_genre.update(distinct_genre)
        all_genre.remove(None)

        item_genre_mat = rated_items[['id', 'genres']].copy()
        item_genre_mat['genres'] = item_genre_mat['genres'].str.lower().str.strip()

        # OHE the genres column
        for genre in all_genre:
            item_genre_mat[genre] = np.where(item_genre_mat['genres'].str.contains(genre), 1, 0)
        item_genre_mat = item_genre_mat.drop(['genres'], axis=1)
        item_genre_mat = item_genre_mat.set_index('id')

        # compute similarity matix
        corr_mat = cosine_similarity(item_genre_mat)

        movies = self.movies.reset_index()
        titles = movies['title']
        indices = pd.Series(movies.index, index=movies['title'])
        if movie_title in titles.values:
            idx = indices[movie_title]
            sim_scores = list(enumerate(corr_mat[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:31]
            movie_indices = [i[0] for i in sim_scores]
            recommendations = titles.iloc[movie_indices].head(10)

            data = movies.merge(recommendations.to_frame(), left_index=True, right_index=True)
            data.rename(columns={'title_x': 'Movie Title', 'genres': 'Genres', 'vote_average': 'Vote Average'}, inplace=True)
            data.reset_index(drop=True, inplace=True)
            return data[['Movie Title', 'Genres', 'Vote Average']]
        else:
            return pd.DataFrame()

