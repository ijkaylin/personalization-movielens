import pandas as pd
import numpy as np
import distance as dist
from collections import namedtuple
import funcs as F
import time

# this is mostly an object wrapper around
# the functional distance.py

# throwing false positives
pd.options.mode.chained_assignment = None

# need complete set to build the map
class KNN:
    def __init__(self, df, complete_set, similarity = 'cosine', by = 'item', k = 5):
        self.df = df # train data, basically
        self.complete_set = complete_set # we need all of it for converting ids
        self.by = by
        self.row, self.col = ('MovieId', 'UserId') if by == 'item' else ('UserId', 'MovieId')
        self.similarity = similarity
        self.k = k

        row_ids = self.complete_set[self.row].unique()
        col_ids = self.complete_set[self.col].unique()

        self.rowix_id_map = dict(zip(range(len(row_ids)), row_ids))
        self.colix_id_map = dict(zip(range(len(col_ids)), col_ids))

        self.rowid_ix_map = { v: k for k, v in self.rowix_id_map.iteritems() }
        self.colid_ix_map = { v: k for k, v in self.colix_id_map.iteritems() }

        # change all user/movie ids, need to undo this when predicting
        self.df['OriginalRow'] = self.df[self.row]
        self.df['OriginalCol'] = self.df[self.col]

        self.df[self.row] = self.df.copy()[self.row].apply(lambda x: self.to_ix(x, by = 'row'))
        self.df[self.col] = self.df.copy()[self.col].apply(lambda x: self.to_ix(x, by = 'col'))

        if (self.similarity == 'adjusted_cosine'):
            self.means = dist.get_means(self.complete_set, 'user')
            self.df = dist.normalize_means(self.df, 'user')
            # from now on pearson is just cosine, and similarity must be item
            self.by = 'item'

        elif (self.similarity == 'pearson'):
            self.means = dist.get_means(self.complete_set, self.by)
            self.df = dist.normalize_means(self.df, self.by)

        else:
            self.means = dist.get_means(self.complete_set, self.by)


    # go back and forth
    # @param original_id
    # @param row or column
    def to_ix(self, id, by = 'row'):
        if by == 'row':
            return self.rowid_ix_map[id]
        else:
            return self.colid_ix_map[id]

    def to_id(self, ix, by = 'row'):
        if by == 'row':
            return self.rowix_id_map[ix]
        else:
            return self.colix_id_map[ix]



    # adjust internal dataframe with normalized means
    # Store the mean centered as the new rating, and preserve
    # the mean rating under `MeanRating`
    def normalize_means(self):
        self.df = dist.normalize_means(self.df, self.by)

    def train(self):
        self.scores = dist.build_similarity_matrix(self.df, self.by, self.similarity)


    # generate predictions, attach them to predictions
    # @return predictions
    def predict(self, new_data = None):
        if new_data is None:
            rm = dist.build_user_item_matrix(self.df, self.by)
            rm = pd.DataFrame.as_matrix(rm)
        else:
            # need to convert to internal ids
            new_data = new_data.copy()
            new_data[self.row] = new_data.copy()[self.row].apply(lambda x: self.to_ix(x, by = 'row'))
            new_data[self.col] = new_data.copy()[self.col].apply(lambda x: self.to_ix(x, by = 'col'))
            rm = dist.build_user_item_matrix(new_data, self.by)
            rm = pd.DataFrame.as_matrix(rm)


        preds = dist.predict(rm, self.scores, self.by, self.k)
        Prediction = namedtuple('Prediction', ['row', 'col', 'estimate', 'actual'])
        # if pearson, need to add the means back
        # @take a Prediction container
        # @return pearson prediction
        def _p(pred):
            row_val = self.to_id(pred.row, 'row')
            if pred.actual is not None:
                actual = pred.actual + self.means[row_val]
            else:
                actual = None

            adj_estimate = pred.estimate + self.means[row_val]
            return Prediction(pred.row, pred.col, adj_estimate, actual)

        # if adjusted cosine, need to add the mean back to actual
        def _ac(pred):
            col_val = self.to_id(pred.col, 'col')
            if pred.actual is not None:
                actual = pred.actual + self.means[col_val]
            else:
                actual = None

            return Prediction(pred.row, pred.col, pred.estimate, actual)


        if self.similarity == 'pearson':
            preds = map(_p, preds)

        return preds


    # compute the MAE
    def mae(self, predictions):
        # just get predictions with actual ratings
        actuals = filter(lambda p: p.actual is not None, predictions)

        n = reduce(lambda a, b: a + abs(b.actual - b.estimate), actuals, 0)
        return n / len(actuals)










