import math
import pandas as pd
import numpy as np
from sr import load_data


class Jaccard:

    def __init__(self):
        self.df_rating = load_data.load_data(load_data.RATINGS, sep=",")

    def fit(self):
        self.df_similarity = load_data.load_data(load_data.JACCARD_SIMILARITY, sep=",")

    def estimate(self, user, item, df_similarity, df_rating):
        result = dict()
        df_similarity = df_similarity[df_similarity['user1']== user].sort_values(by=['sim'], ascending = False)
        df_rating = df_rating[(df_rating['item']== item) & (df_rating['user'].isin(df_similarity['user2']))]
        df_rating = df_rating.merge(df_similarity, left_on='user', right_on='user2')
        prediction = np.dot(df_rating['rating'], df_rating['sim'])/np.sum(df_rating['sim'])
        result['user'] = user
        result['item'] = item
        result['estimation'] = 0 if math.isnan(prediction) else prediction
        result['neighbors'] = list(zip(df_similarity.user2,df_similarity.sim))
        return result

    def predict(self, user, item):
        return self.estimate(user, item, self.df_similarity, self.df_rating)

