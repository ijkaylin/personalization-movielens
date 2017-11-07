import surprise as sp
from surprise import AlgoBase
import pandas as pd
import numpy as np
from sklearn import cross_validation as cv
import funcs as F
import itertools as it
import time

def build_movie_genre_matrix(movies):
    """
    Build a NxM matrix, rows are movie_ids, columns are genres, values 0, 1
    @param movies Dataframe dataframe of movie data
    @returns Matrix like Dataframe with 0s and 1s filled in 
    """
    movie_genre = []
    for (idx, row) in movies.iterrows(): 
        genres = row.loc['Genres'].split("|")
        movieid = row.loc['MovieId']
        for g in genres:  
            movie_genre.append({'MovieId': movieid, 'Genre': g})

    moviegenrecol = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    test = pd.DataFrame(0, index = np.arange(len(movies)), columns = moviegenrecol)
    MovieGenres = pd.concat([movies['MovieId'], test], axis = 1)
    MovieGenres.columns= ['MovieId','Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

    for row in movie_genre: 
        movieID = row['MovieId']
        genre = row['Genre']
        MovieGenres.loc[MovieGenres.MovieId == movieID, genre] = 1

    return MovieGenres

        
def build_user_item_matrix(ratings):
    """
    Return a USERxITEM matrix with values as the user's value for the movie, null otherwise
    Right now not normalized
    @param ratings Dataframe
    @returns matrix numpy matrix with a user's ratings per movie
    """
    matrix = ratings.pivot(index = 'UserId', columns = 'MovieId', values = 'Rating').fillna(0)
    return matrix


def sample(ratings, n, m):
    """
    Return a smaller matrix with top n users and top m items only
    @param ratings the ratings dataset 
    @param n number of users with most ratings
    @param m number of movies with most ratings
    @returns NxM matrix of USERxITEM ratings
    """
    n_users = ratings['UserId'].nunique()
    n_items = ratings['MovieId'].nunique()

    user_sample = ratings['UserId'].value_counts().head(n).index
    movie_sample = ratings['MovieId'].value_counts().head(m).index

    subset = ratings.loc[ratings['UserId'].isin(user_sample)].loc[ratings['MovieId'].isin(movie_sample)]
    # we don't need the timestamp
    del subset['Timestamp']
    return subset


# define our own baseline algorithm that Surprise can use. Just predict the average!
class AvgBase(AlgoBase):
    def __init__(self):
        AlgoBase.__init__(self)

    # see Surprise#building_custom_algo
    def train(self, trainset):
        AlgoBase.train(self, trainset)
        # remember DF is uid, iid, rating
        self.mean = np.mean([rating for (_, _, rating) in self.trainset.all_ratings()])

    def estimate(self, u, i):
        return self.mean


def normalize_user_means(ratings):
    """
    @param Ratings pandas dataframe
    @returns user mean normalized dataframe
    """
    u_i_df = F.build_user_item_matrix(ratings)
    means = u_i_df.mean(axis=1)

    def f(row):
        u_id = row["UserId"]
        new_r = row["Rating"] - means[u_id]
        return new_r

    ratings['Rating'] = ratings.apply(f, axis = 1)
    return ratings
    

# we need something to test our model against. Use above defined model.
def train_baseline(ratings):
    """
    Baseline model. Same as below, return a model and data to test it on
    @param ratings Pandas Dataframe with UserId, MovieId, Ratings
    @returns Tuple (algorithm, testdata) basemodel that just returns the baseline estimate for user/item (adjusted for user bias)
    """
    train_data, test_data = cv.train_test_split(ratings, test_size = 0.20)

    # algo = AvgBase()
    algo = sp.BaselineOnly()
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

    # similarity_options = { 'name': 'pearson', 'user_based': False }
    # algo = sp.KNNWithMeans(sim_options = similarity_options, k = k_neighbors)
    similarity_options = { 'name': 'pearson_baseline', 'user_based': False }
    algo = sp.KNNBaseline(sim_options = similarity_options, k = k_neighbors)

    for trainset, _ in trainset.folds():
        algo.train(trainset)


    testset = testset.build_full_trainset().build_testset()
    return (algo, testset)


def group_predictions_by_user(predictions):
    """
    @param List of Surprise predictions objects
    @returns Dict {uid: [P1, P2, ...PN]} hash mapping user id to top n predictions
    """
    p = sorted(predictions, key = lambda x: x.uid)

    groups = {}
    for k, g in it.groupby(p, lambda x: x.uid):
        groups[k] = sorted(list(g), key = lambda x: x.est, reverse = True)

    return groups


# For every item that a user would be rated, in top k
def calculate_catalog_coverage(ratings, predictions, k):
    """
    Calculate the catalog coverage of a model over a dataset
    @param ratings pandas dataframe with UserId, MovieId, Ratings. Must be the same set the model was trained on
    @param List Surprise predictions
    @oaram k Int the top k recommendations size
    @returns Float percentage of items recommended to at least one user
    """
    n_movies = ratings['MovieId'].nunique()

    movies_reccommended = set() # keep track of which movies are recommended. Note we only care about the number

    recommendations = group_predictions_by_user(predictions)
    for u_id, recs in recommendations.iteritems():
        movies_reccommended.update(map(lambda x: x.iid, recs[0:3]))

    return len(movies_reccommended) / float(n_movies)


# print some evaluative summaries
def evaluate(algo, ratings, testset, top_k = 5):
    """
    @param algo Surprise algorithm the model that was trained
    @oaram ratings The ratings it was trained on, in pandas Dataframe form (so we can calculate coverage)
    @param testset Surprise testset object, the data held out during cross-validation
    @returns Nested Dictionary {test: {rmse, mae}, train: {rmse, mae, cc}}
    We can use these to build up arrays for plotting.
    """

    ret = {}
    ret['test'] = {}
    ret['train'] = {}

    test_predictions = algo.test(testset)
    # see how it would do on the trainset to compare, comes with the algo object
    trainset = algo.trainset.build_testset()
    train_predictions = algo.test(trainset)

    # sticking evaluate in everything for grep, training is verbose
    ret['test']['rmse'] = sp.accuracy.rmse(test_predictions)
    ret['train']['rmse'] = sp.accuracy.rmse(train_predictions)

    ret['test']['mae'] = sp.accuracy.mae(test_predictions)
    ret['train']['mae'] = sp.accuracy.mae(train_predictions)

    # Hackish, baseline does not have a sense of "neighbors"
    if (algo.__module__ == "surprise.prediction_algorithms.knns"):
        ret['test']['cc'] = calculate_catalog_coverage(ratings, test_predictions, top_k)

    return ret




# matrix = funcs.build_user_item_matrix(ratings, movies)
