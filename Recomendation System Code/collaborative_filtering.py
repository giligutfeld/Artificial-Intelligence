# Gili Gutfeld 209284512

import time
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


class Recommender:
    def __init__(self, strategy='user'):
        self.strategy = strategy
        self.similarity = np.NaN

    def fit(self, matrix):
        self.user_item_matrix = matrix
        mean_user_rating = np.nanmean(self.user_item_matrix, axis=1).reshape(-1, 1)

        # Normalize the ratings by subtracting the average of the users ratings
        normalized = (self.user_item_matrix.to_numpy() - mean_user_rating) + 0.001
        normalized[np.isnan(normalized)] = 0

        if self.strategy == 'user':
            # User - User based collaborative filtering
            start_time = time.time()
            self.similarity = 1 - pairwise_distances(normalized, metric='cosine')
            self.pred = pd.DataFrame((mean_user_rating + self.similarity.dot(normalized) /
                                      np.array([np.abs(self.similarity).sum(axis=1)]).T).round(2))

            self.pred.index = matrix.index
            time_taken = time.time() - start_time
            print('User Model in {} seconds'.format(time_taken))
            return self

        elif self.strategy == 'item':
            # Item - Item based collaborative filtering
            start_time = time.time()
            self.similarity = 1 - pairwise_distances(normalized.T, metric='cosine')
            self.pred = pd.DataFrame((mean_user_rating + normalized.dot(self.similarity) /
                                      np.array([np.abs(self.similarity).sum(axis=1)])).round(2))

            self.pred.index = matrix.index
            time_taken = time.time() - start_time
            print('Item Model in {} seconds'.format(time_taken))

            return self

    def recommend_items(self, user_id, k=5):
        try:
            prediction = np.copy(self.pred.loc[user_id].to_numpy())
        except:
            return None
        prediction[~np.isnan(np.copy(self.user_item_matrix.loc[user_id].to_numpy()))] = 0
        indexs = np.argsort(-prediction, kind='mergesort')
        scores = indexs[0:k]

        top_k_similar_items = []
        for s in scores:
            top_k_similar_items.append(self.user_item_matrix.columns[s])
        return top_k_similar_items
