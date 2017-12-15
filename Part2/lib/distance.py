import pandas as pd
import numpy as np
import random as rand
import funcs as F
from collections import namedtuple
import time


toy = pd.read_csv('../toy.dat', sep = '::', names = ['UserId', 'MovieId', 'Rating', 'Timestamp'], engine = 'python')
# _100k = pd.read_csv('../ratings100k.dat', sep = '::', names = ['UserId', 'MovieId', 'Rating', 'Timestamp'], engine = 'python')
# ratings = pd.read_csv('../ratings.dat', sep = '::', names = ['UserId', 'MovieId', 'Rating', 'Timestamp'], engine = 'python')

# @param v vector of numerics
# @param w vector of numerics
# @returns Float dot product of two vectors
# @raises Exception if vectors are not of the same length
def dot_product(v, w):
    if len(v) != len(w):
        raise Exception("Vectors not same length")


    ret = np.dot(v, w)

    # ret = 0
    # for i in range(0, len(v)):
    #     ret += (v[i] * w[i])

    return ret

# doesn't really need to be a function, but makes it more obvious if we forget to subtract when indexing the Matrix
# id to index
# @param Movie or UserId
# @returns Matrix column or row index
def to_ix(n):
    return n-1

def to_id(n):
    return n+1

# @param v vector
# @returns Float the length of the vector
def magnitude(v):
    return dot_product(v, v) ** 0.5


# cosine between vectors, metric of similarity
# @param v vector of numerics
# @param v vector of numerics
# @returns Float cosine of the angle between the vectors
def cosine(v, w):
    dp = dot_product(v, w)
    len1 = magnitude(v)
    len2 = magnitude(w)

    # no division by 0
    if (len1 == 0 or len2 == 0):
        return 0

    return dp / (len1 * len2)


# get common ratings for two users (or items, if item based)
# ratings should be normalized before calling this function
# @param String|Int UserOrItemId
# @param String|Int UserOrItemId
# @param String { 'user', 'item' }
# @returns (v1, v2) that can have a similarity metric run on them
def get_common_ratings(df1, df2, by = 'item'):
    join = 'UserId' if by == 'item' else 'MovieId'

    c = pd.merge(df1, df2, how = 'inner', on=[join])

    v1 = list(c.Rating_x)
    v2 = list(c.Rating_y)
    return (v1, v2)


# @param df ratings dataframe
# @param String|Int UserOrItemId
# @param String|Int UserOrItemId
# @param String { 'user', 'item' }
# @returns Float the cosine score between u, v
def cosine_score(df1, df2, by = 'item'):
    r1, r2 = get_common_ratings(df1, df2, by)
    return cosine(r1, r2)


# @param df ratings dataframe
# @param String|Int UserOrItemId
# @param String|Int UserOrItemId
# @param String { 'user', 'item' }
# @returns Float the Pearson score between u, v
def pearson_score(df, u, v, by = 'item'):
    # normalize means first
    df = normalize_means(df, by = by)
    r1, r2 = get_common_ratings(df, u, v, by)
    return cosine(r1, r2)


# @param ratings Dataframe
# @param by = { 'user', 'item' }
# @param String similarity type { "cosine", "adjusted_cosine", "pearson" }
# @returns NumPy Matrix with similarity scores
def build_similarity_matrix(df, by = 'user', similarity = 'cosine'):
    field = 'UserId' if by == 'user' else 'MovieId'
    unique_ids = df[field].unique()
    dim = len(unique_ids)
    scores = np.full((dim, dim), 0, dtype=np.float64)

    to_score = (dim * dim) / 2.0
    scored = 0

    print "There are {} unique ids, so O({}) items to score".format(dim, to_score)

    # build up user/item subsets, so we don't need to do this over and over again
    subsets = {}
    print 'Starting to build subsets'
    for i in range(0, dim):
        id = to_id(i)
        subsets[id] = df.loc[df[field] == id]

    print 'Finished building subsets'

    start = time.time()
    for i in range(0, dim):
        id_1 = to_id(i)
        df1 = subsets[id_1]
        for j in range(i+1, dim):
            id_2 = to_id(j)
            df2 = subsets[id_2]

            score = cosine_score(df1, df2, by)
            scores[i, j] = score
            scores[j, i] = score

            scored += 1
            if (scored % 1000 == 0):
                elapsed = round(time.time() - start)
                print "Finished scoring {}/{} items, {} percent finished, {} seconds".format(scored, to_score, round(( (scored / to_score) * 100 ), 2), elapsed )

    return scores


# predict a rating for user, item combo
# NB Scores and ratings should be appropriately transposed already
# @param ratings a ratings matrix, not the original ratings dataframe
# @param scores matrix Similarity Matrix
# @param row_id Either user or item id, depending on what it's based on. We want the base in the rows
# @param col_id String|Int item_id
# @param k number of user/items to compare to
# @returns Dict { 'user_id' : [] }
def predict_user_item_pair(rm, scores, row_id, col_id, k = 5):
    # turn the ids into matrix_indices
    print "Predicting {}, {}".format(u_id, i_id)
    row_id = to_ix(u_id)
    col_id = to_ix(i_id)

    num_cols = rm.shape[1]

    # for the u_id, get the n most similar things, shove in (i_id, score, rating) tuples, can sort later
    top_k = []
    for i in range(0, num_cols):
        if rm[row_id, i] != 0:
            sim = scores[row_id, i]
            rating = rm[row_id, i]
            top_k.append((to_id(i), sim, rating))

    top_k = sorted(top_k, key = lambda x: x[1], reverse = True)
    top_k = top_k[:k]

    n = reduce(lambda a, b: a + (b[2] * b[1]), top_k, 0)
    d = reduce(lambda a, b: a + b[1], top_k, 0)

    if d == 0: return 0
    return n / d



# predict all
# @returns List{Prediction Tuples}
def predict(rm, scores, by = 'item', k = 5):
    print "Calling predict"
    row, col = ('i_id', 'u_id') if by == 'item' else ('u_id', 'i_id')
    Prediction = namedtuple('Prediction', [row, col, 'estimate', 'actual'])

    if by == 'item':
        _rm = rm.transpose()
        _scores = scores.transpose()

    _len, _width = rm.shape
    print _len, _width

    finished = 0
    to_finish = _len * _width

    ret = []
    for r in range(0, _len):
        for  c in range(0, _width):
            row_id = to_id(r)
            col_id = to_id(c)

            estimate = predict_user_item_pair(rm, scores, row_id, col_id)
            actual = rm[r, c] if rm[r, c] != 0 else None

            pred = Prediction(row_id, col_id, estimate, actual)
            ret.append(pred)

            finished += 1
            if finished % 1000 == 0:
                print "Finished predicting {}/{}".format(finished, to_finish)

    return ret


def build_user_item_matrix(df):
    """
    Return a USERxITEM matrix with values as the user's value for the movie, null otherwise
    Right now not normalized
    @param ratings Dataframe
    @returns matrix numpy matrix with a user's ratings per movie
    """
    matrix = df.pivot(index = 'UserId', columns = 'MovieId', values = 'Rating').fillna(0)
    return matrix


# @param ratings Dataframe
# @param by String { 'user', 'item' }
# @returns Dataframe with same structure but normalized means
def normalize_means(df, by = 'item'):
    mt = build_user_item_matrix(df)
    if (by != 'user'):
        mt = mt.transpose()

    # handle filled in 0s
    means = np.true_divide(mt.sum(axis=1), (mt!=0).sum(axis=1))
    row, col = ('MovieId', 'UserId') if by == 'item' else ('UserId', 'MovieId')

    def f(_row):
        row_id = _row[row]
        normalized = _row['Rating'] - means[row_id]
        return normalized

    df['Rating'] = df.apply(f, axis = 1)
    return df



# build similarity matrix, time it
# we can subset the data
# user_counts = ratings['UserId'].value_counts()
# movie_counts = ratings['MovieId'].value_counts()

# subset = F.sample(ratings, user_counts, movie_counts, 30000, 1000)

# start = time.time()
# sims = build_similarity_matrix(subset, by = 'item', similarity = 'cosine')
# end = time.time()
# print "elapsed: {}".format(end - start)

# print "Writing similarity scores to file"
# np.save('../similarity.txt', arr = sims)
