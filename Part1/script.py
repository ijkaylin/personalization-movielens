# Test some surprise stuff

import surprise as sp
from surprise import AlgoBase
import pandas as pd
import numpy as np
from sklearn import cross_validation as cv
import funcs as F

moviescol = ['MovieId', 'Title', 'Genres','Action', 'Adventure',
 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

_ratings = pd.read_csv('./ratings1m.dat', sep = '::', names = ['UserId', 'MovieId', 'Rating', 'Timestamp'], engine = 'python')
_movies = pd.read_csv('./movies.dat', sep ='::', names = moviescol, engine='python')


# small = F.sample(_ratings, 100, 10) # subset of 100 most popular movies users, 10 most popular movies
# medium = F.sample(_ratings, 1000, 50)
# large = F.sample(_ratings, 5000, 100)

print "Evaluate"

# define our own baseline algorithm that Surprise can use. Just predict the average!
class AvgBase(AlgoBase):
    def __init__(self):
        AlgoBase.__init__(self)

    # see Surprise#building_custom_algo
    def train(self, trainset):
        AlgoBase.train(self, trainset)
        # remember DF is uid, iid, rating
        self.mean = np.mean([rating for (_, _, rating) in self.trainset.all_ratings()])
        print "Evaluate: Mean in base is {}".format(self.mean)

    def estimate(self, u, i):
        return self.mean

# we need something to test our model against. Use above defined model.
def train_baseline(ratings):
    """
    Baseline model. Same as below, return a model and data to test it on
    @param ratings Pandas Dataframe with UserId, MovieId, Ratings
    @returns Tuple (algorithm, testdata) basemodel that just returns the baseline estimate for user/item (adjusted for user bias)
    """
    train_data, test_data = cv.train_test_split(ratings, test_size = 0.20)

    algo = AvgBase()
    reader = sp.Reader(rating_scale=(1, 5))

    trainset = sp.Dataset.load_from_df(ratings, reader)
    testset = sp.Dataset.load_from_df(test_data, reader)

    trainset = trainset.build_full_trainset()

    algo.train(trainset)

    testset = testset.build_full_trainset().build_testset()
    return (algo, testset)



# train the KNN model on subsets of the data (for cross-validation)
def train(ratings, k_neighbors, k_folds):
    """
    Train a model and return it. Then we can use the model and evaluate it elsewhere
    @param ratings dataframe pandas dataframe to train on, with columns UserId, MovieId, Ratings
    @param k_neighbors number of neighbors to examine
    @param k_folds number of folds for cross validation
    @returns List of (algo, test data)
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


def calculate_catalog_coverage(ratings, algo, k):
    """
    Calculate the catalog coverage of a model over a dataset
    @param ratings pandas dataframe with UserId, MovieId, Ratings. Must be the same set the model was trained on
    @param algo Surprise KNN algorithm which has ALREADY been trained; we can use this to find neighbors
    @oaram k Int the neighborhood size
    @returns Float percentage of items recommended to at least one user
    """
    movie_ids = ratings['MovieId'].unique()
    n_movies = len(movie_ids)

    movies_reccommended = set() # keep track of which movies are recommended. Note we only care about the number
    for m_id in movie_ids:
        # note, we need the surprise internal `inner_id` to query neighbors
        inner_id = algo.trainset.to_inner_iid(m_id)
        neighbors = algo.get_neighbors(inner_id, k)
        for n in neighbors:
            raw_id = algo.trainset.to_raw_iid(n)
            movies_reccommended.add(raw_id)

    return len(movies_reccommended) / float(n_movies)


# print some evaluative summaries
def evaluate(algo, ratings, testset):
    """
    Print some u
    @param algo Surprise algorithm the model that was trained
    @oaram ratings The ratings it was trained on, in pandas Dataframe form (so we can calculate coverage)
    @param testset Surprise testset object, the data held out during cross-validation
    """
    test_predictions = algo.test(testset)
    # see how it would do on the trainset to compare, comes with the algo object
    trainset = algo.trainset.build_testset()
    train_predictions = algo.test(trainset)

    # sticking evaluate in everything for grep, training is verbose
    print "Evaluate: RMSE of the testset is {}".format(sp.accuracy.rmse(test_predictions))
    print "Evaluate: RMSE of the trainset is {}".format(sp.accuracy.rmse(train_predictions))

    print "Evaluate: MAE of the testset is {}".format(sp.accuracy.mae(test_predictions))
    print "Evaluate: MAE of the trainset is {}".format(sp.accuracy.mae(train_predictions))

    # Hackish, baseline does not have a sense of "neighbors"
    if (algo.__module__ == "surprise.prediction_algorithms.knns"):
        print "Evaluate: CC of the model is {}".format(calculate_catalog_coverage(ratings, algo, algo.k))



# run models with some different parameters and sizes
samples = [ [1000, 10], [5000, 50], [100000, 1000], [5000000, 2000] ]
k_s = [3, 5, 10, 15, 30, 40]

for sample in samples:
    i, j = sample
    _dataset = F.sample(_ratings, i, j)
    print "Evaluating Baseline and KNN on the dataset with {} users and {} items".format(i, j)
    print "Evaluating baseline"

    base, base_test = train_baseline(_dataset)
    evaluate(base, _dataset, base_test)

    for k in k_s:
        knn, knn_test = train(_dataset, k, 5)

        print "Evaluating KNN with k of {}".format(k)
        evaluate(knn, _dataset, knn_test)


