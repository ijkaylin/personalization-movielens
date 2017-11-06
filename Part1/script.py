# Test some surprise stuff

import surprise as sp
import pandas as pd
from sklearn import cross_validation as cv
import funcs as F

moviescol = ['MovieId', 'Title', 'Genres','Action', 'Adventure',
 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

_ratings = pd.read_csv('./ratings100k.dat', sep = '::', names = ['UserId', 'MovieId', 'Rating', 'Timestamp'], engine = 'python')
_movies = pd.read_csv('./movies.dat', sep ='::', names = moviescol, engine='python')


small_subset = F.sample(_ratings, 100, 10) # subset of 100 most popular movies users, 10 most popular movies
# print small_subset

# # load_from_df requires 3 columns, NOT a matrix, user_ids, item_ids, ratings, in order
# dataset = sp.Dataset.load_from_df(small_subset, sp.Reader(rating_scale=(1,5)))
# trainset = dataset.build_full_trainset()

# similarity_options = { 'name': 'pearson', 'user_based': False }
# algo = sp.KNNWithMeans(sim_options = similarity_options, k = 5)

# algo.train(trainset)


# train the KNN model on subsets of the data (for cross-validation)
def train(ratings, k_neighbors, k_folds):
    """
    Train a model and return it. Then we can use the model and evaluate it elsewhere
    @param ratings dataframe pandas dataframe to train on, with columns UserId, MovieId, Ratings
    @param k_neighbors number of neighbors to examine
    @param k_folds number of folds for cross validation
    @returns List of (algo, remaining test fold) tuples a surprise.KNNWithMeans a Nearest Neigbbor algorithm and its predicitons object from Surprise library
    We can call methods such as `test` and `evaluate` on this object 
    """

    train_data, test_data = cv.train_test_split(ratings, test_size = 0.20)
    reader = sp.Reader(rating_scale=(1, 5))

    trainset = sp.Dataset.load_from_df(train_data, reader)
    testset = sp.Dataset.load_from_df(test_data, reader)
    trainset.split(n_folds = k_folds)

    similarity_options = { 'name': 'pearson', 'user_based': False }
    algo = sp.KNNWithMeans(sim_options = similarity_options, k = k_neighbors)

    for trainset, _ in trainset.folds():
        algo.train(trainset)


    testset = testset.build_full_trainset().build_testset()
    return (algo, testset)


# return some evaluation summaries
def evaluate(algo, testset):
    predictions = algo.test(testset)
    print "RMSE is {}".format(sp.accuracy.rmse(predictions))
    return True



