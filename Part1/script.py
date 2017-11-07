# # Test some surprise stuff

# import surprise as sp
# from surprise import AlgoBase
# import numpy as np
# from sklearn import cross_validation as cv
# import itertools as it
import time
import pandas as pd
import pprint as pp
import funcs as F

moviescol = ['MovieId', 'Title', 'Genres','Action', 'Adventure',
 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

print "Evaluate: Starting the script"

_ratings = pd.read_csv('./ratings100k.dat', sep = '::', names = ['UserId', 'MovieId', 'Rating', 'Timestamp'], engine = 'python')
_movies = pd.read_csv('./movies.dat', sep ='::', names = moviescol, engine='python')

samples = [ [1000, 100], [5000, 500]] # , [100000, 4000] ]
k_s = [3, 5, 10, 15, 30, 40]
top_k = 5
all_results = []

for sample in samples:
    i, j = sample
    _dataset = F.sample(_ratings, i, j)
    print "Evaluating Baseline and KNN on the dataset with {} users and {} items".format(i, j)
    print "Evaluating baseline"

    base, base_test = F.train_baseline(_dataset)
    base_eval = F.evaluate(base, _dataset, base_test)
    all_results.append(base_eval)

    for k in k_s:
        # rough timer, not great at measuring v.fast computations but probably ok for us
        t_0 = time.time()

        knn, knn_test = F.train(_dataset, k, 5)

        print "Evaluating KNN with k of {}".format(k)
        results = F.evaluate(knn, _dataset, knn_test, top_k)
        # add k, and sample size to results
        results['sample'] = sample
        results['k'] = k

        # add a rough time measurement
        t_1 = time.time()
        elapsed = t_1 - t_0
        results['time'] = elapsed 

        all_results.append(results)

_print = pp.PrettyPrinter(depth = 2, indent = 2)
for results in all_results:
    _print.pprint(results)


