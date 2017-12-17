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
    # normalize cosine similarity b/t 0, 1
    sim = cosine(r1, r2)
    cos_distance = (1 - sim) / 2
    return 1 - cos_distance


# @param df ratings dataframe, with means normalized
# @param String|Int UserOrItemId
# @param String|Int UserOrItemId
# @param String { 'user', 'item' }
# @returns Float the Pearson score between u, v
def pearson_score(df1, df2, by = 'item'):
    # normalize means first
    r1, r2 = get_common_ratings(df1, df2, by)
    return cosine(r1, r2)


# @param ratings Dataframe
# @param by = { 'user', 'item' }
# @param String similarity type { "cosine", "adjusted_cosine", "pearson" }
# @returns NumPy Matrix with similarity scores
def build_similarity_matrix(df, by = 'item', similarity = 'cosine'):
    field = 'MovieId' if by == 'item' else 'UserId'
    unique_ids = df[field].unique()
    dim = len(unique_ids)
    scores = np.full((dim, dim), 0, dtype=np.float64)

    to_score = (dim * dim) / 2.0
    scored = 0

    func = cosine_score
    if similarity == 'pearson':
        func = pearson_score

    print "There are {} unique ids, so O({}) items to score".format(dim, to_score)

    # build up user/item subsets, so we don't need to do this over and over again
    subsets = {}
    print 'Starting to build subsets'
    for i in range(0, dim):
        subsets[i] = df.loc[df[field] == i]

    print 'Finished building subsets'

    start = time.time()
    for i in range(0, dim):
        df1 = subsets[i]
        for j in range(i+1, dim):
            df2 = subsets[j]

            score = func(df1, df2, by)
            scores[i, j] = score
            scores[j, i] = score

            scored += 1
            if (scored % 1000 == 0):
                elapsed = round(time.time() - start)
                print "Finished scoring {}/{} items, {} percent finished, {} seconds".format(scored, to_score, round(( (scored / to_score) * 100 ), 2), elapsed )

    return scores


# predict a rating for user, item combo
# NB Scores and ratings should be appropriately transposed already
# NB row_id, col_id should be raw ids
# @param ratings a ratings matrix, not the original ratings dataframe
# @param scores matrix Similarity Matrix
# @param row_id Either user or item id, depending on what it's based on. We want the base in the rows
# @param col_id String|Int item_id
# @param k number of user/items to compare to
# @returns Dict { 'user_id' : [] }
def predict_user_item_pair(rm, scores, row_id, col_id, k = 5):
    num_rows = rm.shape[0]

    # for the u_id, get the n most similar things, shove in (i_id, score, rating) tuples, can sort later
    top_k = []
    for i in range(0, num_rows):
        if rm[i, col_id] != 0:
            sim = scores[row_id, i]
            rating = rm[i, col_id] # this is an item/user rating for item based
            top_k.append((i, sim, rating))

    top_k = sorted(top_k, key = lambda x: x[1], reverse = True)
    top_k = top_k[:k]

    n = reduce(lambda a, b: a + (b[2] * b[1]), top_k, 0)
    d = reduce(lambda a, b: a + b[1], top_k, 0)

    if d == 0: return 0
    return n / d



# predict all
# @param List{(row_id, col_id)} to look up
# @returns List{Prediction Tuples}
def predict(rm, scores, row_cols, by = 'item', k = 5):
    Prediction = namedtuple('Prediction', ['row', 'col', 'estimate', 'actual'])

    _len = len(row_cols)
    print "Len pre filter: {}".format(_len)
    # filter row_cols
    print "shape: {}".format(rm.shape)
    max_r, max_c = rm.shape
    max_r, max_c = max_r - 1, max_c - 1

    print "max_r, max_c => {}, {}".format(max_r, max_c)

    row_cols = filter(lambda t: t[0] <= max_r and t[1] <= max_c, row_cols)
    _len = len(row_cols)

    max_col = max(row_cols, key = lambda x: x[1])[1]
    print "max_col => {}".format(max_col)

    print "Len post filter: {}".format(_len)

    finished = 0
    to_finish = _len


    ret = []
    for r, c in row_cols:
        if rm[r, c] == 0:
            finished += 1
            continue

        actual = rm[r, c]
        estimate = predict_user_item_pair(rm, scores, r, c, k)

        pred = Prediction(r, c, estimate, actual)
        ret.append(pred)

        finished += 1
        if finished % 1000 == 0:
            print "Finished predicting {}/{}".format(finished, to_finish)

    return ret


def build_user_item_matrix(df, by = 'item'):
    """
    Return a USERxITEM matrix with values as the user's value for the movie, null otherwise
    Right now not normalized
    @param ratings Dataframe
    @returns matrix numpy matrix with a user's ratings per movie
    """

    index, columns = ('UserId', 'MovieId') if by != 'item' else ('MovieId', 'UserId') 
    matrix = df.pivot(index = index, columns = columns, values = 'Rating').fillna(0)
    return matrix


# @param ratings Dataframe
# @param by String { 'user', 'item' }
# @attach means
# @returns Dataframe with same structure but normalized means
def normalize_means(df, by = 'item'):
    means = get_means(df, by)
    row, col = ('MovieId', 'UserId') if by == 'item' else ('UserId', 'MovieId')

    def f(_row):
        row_id = _row[row]
        normalized = _row['Rating'] - means[row_id]
        return normalized

    def g(_row):
        row_id = _row[row]
        return means[row_id]

    df['MeanRating'] = df.apply(g, axis = 1)
    df['Rating'] = df.apply(f, axis = 1)
    return df


def get_means(df, by = 'item'):
    mt = build_user_item_matrix(df, by)

    # handle 0s
    np.seterr(divide = 'ignore', invalid = 'ignore')
    means = np.true_divide(mt.sum(axis=1), (mt!=0).sum(axis=1))
    return means



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
