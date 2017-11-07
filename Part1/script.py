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
import json

moviescol = ['MovieId', 'Title', 'Genres','Action', 'Adventure',
 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

print "Starting the script"

_ratings = pd.read_csv('./ratings1m.dat', sep = '::', names = ['UserId', 'MovieId', 'Rating', 'Timestamp'], engine = 'python')
_movies = pd.read_csv('./movies.dat', sep ='::', names = moviescol, engine='python')

# samples = [ [10000, 100], [15000, 200] , [15000, 500], [30000, 2000] ]
samples = [ [5000, 100], [10000, 200] ]

k_s = range(5, 60, 5)
factor_sizes = range(5, 60, 5)
top_k = 5
all_results = []

for sample in samples:
    i, j = sample
    _dataset = F.sample(_ratings, i, j)
    print "Running Baseline, MF, KNN on the dataset with {} users and {} items".format(i, j)

    base, base_test = F.train_baseline(_dataset)
    base_eval = F.evaluate(base, _dataset, base_test)
    base_eval['name'] = 'baseline'

    all_results.append(base_eval)

    for f in factor_sizes:
        t_0 = time.time()

        mf, mf_test = F.train_matrix(_dataset, f, 5)

        print "Running MF with F of {}".format(f)
        results = F.evaluate(mf, _dataset, mf_test, top_k)
        # add k, and sample size to results
        results['sample'] = sample
        results['f'] = f

        # add a rough time measurement
        t_1 = time.time()
        elapsed = t_1 - t_0
        results['time'] = elapsed
        results['name'] = 'mf'

        all_results.append(results)

    for k in k_s:
        # rough timer, not great at measuring v.fast computations but probably ok for us
        t_0 = time.time()

        knn, knn_test = F.train(_dataset, k, 5)

        print "Running KNN with k of {}".format(k)
        results = F.evaluate(knn, _dataset, knn_test, top_k)
        # add k, and sample size to results
        results['sample'] = sample
        results['k'] = k

        # add a rough time measurement
        t_1 = time.time()
        elapsed = t_1 - t_0
        results['time'] = elapsed
        results['name'] = 'knn'

        all_results.append(results)

_print = pp.PrettyPrinter(depth = 2, indent = 2)
for results in all_results:
    _print.pprint(results)

    json = json.dumps(results)
    f = open('./out/eval.json', 'w')
    f.write(json)
    f.close()



