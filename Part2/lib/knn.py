import pandas as pd
import numpy as np
import distance as dist
from collections import namedtuple

# this is mostly an object wrapper around
# the functional distance.py

# throwing false positives
pd.options.mode.chained_assignment = None

# need complete set to build the map
class KNN():
    def __init__(self, df, complete_set, similarity = 'cosine', by = 'item', k = 5):
        self.df = df # train data, basically
        self.complete_set = complete_set # we need all of it for converting ids
        self.by = by
        self.row, self.col = ('MovieId', 'UserId') if by == 'item' else ('UserId', 'MovieId')

        # convert to internal ids
        self.build_id_maps()
        self.df = self.convert_ids(self.df)
        print "Finished converting ids"

        self.similarity = similarity
        self.k = k
        # normalize by the *user*, but still do everything
        # else by item
        if (similarity == 'adjusted_cosine'):
            self.means = dist.get_means(self.df, 'user')
            self.df = dist.normalize_means(self.df, 'user')
            # from now on pearson is just cosine, and similarity must be item
            self.similarity = 'cosine'
            self.by = 'item'

        elif (similarity == 'pearson'):
            self.means = dist.get_means(self.df, by)
            self.df = dist.normalize_means(self.df, self.by)

        else:
            self.means = dist.get_means(self.df, by)

        print self.means



    # we want ids to go from 0 -> length, for matrix operations 
    # but we need to be able to get back the originals
    def build_id_maps(self):
        row_ids = self.complete_set[self.row].unique()
        col_ids = self.complete_set[self.col].unique()

        self.rowix_id_map = dict(zip(range(len(row_ids)), row_ids))
        self.colix_id_map = dict(zip(range(len(col_ids)), col_ids))

        self.rowid_ix_map = { v: k for k, v in self.rowix_id_map.iteritems() }
        self.colid_ix_map = { v: k for k, v in self.colix_id_map.iteritems() }

        print self.rowid_ix_map
        print self.colid_ix_map


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

    def convert_ids(self, _df):
        # change all user/movie ids, need to undo this when predicting
        _df['OriginalRow'] = _df[self.row]
        _df['OriginalCol'] = _df[self.col]

        _df[self.row] = _df.apply(lambda row: self.to_ix(row[self.row], 'row'), axis = 1)
        _df[self.col] = _df.apply(lambda row: self.to_ix(row[self.col], 'col'), axis = 1)

        return _df



    # adjust internal dataframe with normalized means
    # Store the mean centered as the new rating, and preserve
    # the mean rating under `MeanRating`
    def normalize_means(self):
        self.df = dist.normalize_means(self.df, self.by)

    def train(self):
        print "Computing similarity matrix"
        self.scores = dist.build_similarity_matrix(self.df, self.by, self.similarity)


    # generate predictions, attach them to predictions
    # @return predictions
    def predict(self, new_data = None):
        print "Computing predictions"

        if new_data is None:
            rm = dist.build_user_item_matrix(self.df, self.by)
            rm = pd.DataFrame.as_matrix(rm)
        else:
            # need to convert to internal ids
            new_data = self.convert_ids(new_data)
            rm = dist.build_user_item_matrix(new_data, self.by)
            rm = pd.DataFrame.as_matrix(rm)


        preds = dist.predict(rm, self.scores, self.by, self.k)
        Prediction = namedtuple('Prediction', ['row', 'col', 'estimate', 'actual'])
        # if pearson, need to add the means back
        # @take a Prediction container
        # @return pearson prediction
        def _p(pred):
            if pred.actual is not None:
                actual = pred.actual + self.means[pred.row]
            else:
                actual = None

            adj_estimate = pred.estimate + self.means[pred.row]
            return Prediction(pred.row, pred.col, adj_estimate, actual)

        # if adjusted cosine, need to add the mean back to actual
        def _ac(pred):
            if pred.actual is not None:
                actual = pred.actual + self.means[pred.col]
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


