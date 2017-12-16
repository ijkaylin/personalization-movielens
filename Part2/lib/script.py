import distance as dist
import funcs as F
import numpy as np
import pandas as pd
import knn as knn
import json
import pprint as pp
import time


toy = pd.read_csv('../toy.dat', sep = '::', names = ['UserId', 'MovieId', 'Rating', 'Timestamp'], engine = 'python')
ratings = pd.read_csv('../ratings10k.dat', sep = '::', names = ['UserId', 'MovieId', 'Rating', 'Timestamp'], engine = 'python')
# ratings = pd.read_csv('../ratings.dat', sep = '::', names = ['UserId', 'MovieId', 'Rating', 'Timestamp'], engine = 'python')



# knn = knn.KNN(subset, similarity = 'pearson', by = 'item', k = 5)
# knn.train()
# preds = knn.predict()


# samples = [ [5000, 100], [10000, 200], [15000, 500] ]

user_counts = ratings['UserId'].value_counts()
movie_counts = ratings['MovieId'].value_counts()
subset = F.sample(ratings, user_counts, movie_counts, 500, 50)

# @param take a pandas dataframe, sample it
def train_test(df):
    msk = np.random.rand(len(df)) < 0.8
    train = df[msk]
    test = df[~msk]

    return (train, test)

samples = [ [500, 10], [1000, 20], [1500, 50] ]
distances = ['pearson', 'adjusted_cosine']
all_results = []
k_s = range(5, 15, 5)
# factor_sizes = range(5, 45, 5)

for sample in samples:
    i, j = sample
    subset = F.sample(ratings, user_counts, movie_counts, i, j)
    train, test = train_test(subset)

    print "Running Baseline, KNN on the dataset with {} users and {} items".format(i, j)

    base, base_test = F.train_baseline(subset)
    base_eval = F.evaluate(base, subset, base_test)
    base_eval['name'] = 'baseline'
    base_eval['sample'] = sample

    all_results.append(base_eval)


    for k in k_s:
        for d in distances:
        # rough timer, not great at measuring v.fast computations but fine for us
            t_0 = time.time()
            _knn = knn.KNN(train, ratings, similarity = d, by = 'item', k = k)
            _knn.train()

            train_preds = _knn.predict()
            test_preds = _knn.predict(test)

            results = {}
            results['sample'] = sample
            results['name'] = 'knn'
            results['distance'] = d

            print "Running KNN with k of {}".format(k)
            results = F.evaluate(knn, _dataset, knn_test, top_k)
            # add k, and sample size to results
            results['sample'] = sample
            results['k'] = k

            results['test'] = { 'mae': 0, 'cc': 0.75 }
            results['train'] = { 'mae': 0 , 'cc': 0.75 }

            results['train']['mae'] = _knn.mae(train_preds)
            results['test']['mae'] = _knn.mae(test_preds)

            # add a rough time measurement
            t_1 = time.time()
            elapsed = t_1 - t_0
            results['time'] = elapsed

            all_results.append(results)


_print = pp.PrettyPrinter(depth = 2, indent = 2)
for results in all_results:
    _print.pprint(results)

_json = json.dumps(all_results)
f = open('../out/output.json', 'w')
f.write(_json)
f.close()
