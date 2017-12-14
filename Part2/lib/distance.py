import pandas as pd
import numpy as np
import random as rand

toy = pd.read_csv('../toy.dat', sep = '::', names = ['UserId', 'MovieId', 'Rating', 'Timestamp'], engine = 'python')

# @param v vector of numerics
# @param w vector of numerics
# @returns Float dot product of two vectors
# @raises Exception if vectors are not of the same length
def dot_product(v, w):
    if len(v) != len(w):
        raise Exception("Vectors not same length")

    ret = 0
    for i in range(0, len(v)):
        ret += (v[i] * w[i])

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
    return dp / (len1 * len2)


# get common ratings for two users (or items, if item based)
# ratings should be normalized before calling this function
# @param df ratings dataframe
# @param String|Int UserOrItemId
# @param String|Int UserOrItemId
# @param String { 'user', 'item' }
# @returns (v1, v2) that can have a similarity metric run on them
def get_common_ratings(df, u, v, by = 'item'):
    field, join = ('MovieId', 'UserId') if by == 'item' else ('UserId', 'MovieId')

    df1 = df.loc[df[field] == u]
    df2 = df.loc[df[field] == v]

    c = pd.merge(df1, df2, how = 'inner', on=[join])

    v = list(c.Rating_x)
    w = list(c.Rating_y)
    return (v, w)


# @param df ratings dataframe
# @param String|Int UserOrItemId
# @param String|Int UserOrItemId
# @param String { 'user', 'item' }
# @returns Float the cosine score between u, v
def cosine_score(df, u, v, by = 'item'):
    r1, r2 = get_common_ratings(df, u, v, by)
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
    row_name = 'UserId' if by == 'user' else 'MovieId'
    unique_ids = df[row_name].unique()
    dim = len(unique_ids)
    scores = np.full((dim, dim), 0, dtype=np.float64)

    for i in range(0, dim):
        for j in range(i+1, dim):
            id_1 = to_id(i)
            id_2 = to_id(j)

            print id_1, id_2
            score = cosine_score(df, id_1, id_2, by)
            print score
            scores[i, j] = score
            scores[j, i] = score

    return scores


def build_user_item_matrix(df):
    """
    Return a USERxITEM matrix with values as the user's value for the movie, null otherwise
    Right now not normalized
    @param ratings Dataframe
    @returns matrix numpy matrix with a user's ratings per movie
    """
    matrix = ratings.pivot(index = 'UserId', columns = 'MovieId', values = 'Rating').fillna(0)
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
