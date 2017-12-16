import distance as dist
import funcs as F
import numpy as np
import pandas as pd
import knn as knn
import json
import pprint as pp
import time
import gc


toy = pd.read_csv('../toy.dat', sep = '::', names = ['UserId', 'MovieId', 'Rating', 'Timestamp'], engine = 'python')
ratings = pd.read_csv('../ratings100k.dat', sep = '::', names = ['UserId', 'MovieId', 'Rating', 'Timestamp'], engine = 'python')
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


# samples = [ [5000, 100], [10000, 200], [15000, 500] ]
samples = [ [500, 10], [1000, 20], [1500, 50] ]
all_results = []
k_s = range(5, 45, 5)
factor_sizes = range(5, 45, 5)

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

        # MF model
        t_0 = time.time()

        mf, mf_test = F.train_matrix(subset, k, 5)

        print "Running MF with k of {}".format(k)
        results = F.evaluate(mf, subset, mf_test, k)
        # add k, and sample size to results
        results['sample'] = sample
        results['f'] = k

        # add a rough time measurement
        t_1 = time.time()
        elapsed = t_1 - t_0
        results['time'] = elapsed
        results['name'] = 'mf'

        # finally, we need these for the hybrid
        trainset = mf.trainset.build_testset()
        anti_trainset = mf.trainset.build_anti_testset()

        train_preds = mf.test(trainset)
        anti_preds = mf.test(anti_trainset)

        mf_preds = train_preds + anti_preds

        all_results.append(results)



        # KNN model
        # t_0 = time.time()
        # print "Running KNN with k of {}".format(k)

        # algo = F.custom_knn(train.copy(), subset.copy(), similarity = 'adjusted_cosine', by = 'item', k = k)

        # train_preds = algo.predict()
        # test_preds = algo.predict(test.copy())

        # results = {}
        # results['sample'] = sample
        # results['name'] = 'knn'

        # results['k'] = k

        # results['test'] = { 'mae': 0, 'cc': 0.75 }
        # results['train'] = { 'mae': 0 , 'cc': 0.75 }

        # results['train']['mae'] = algo.mae(train_preds)
        # results['test']['mae'] = algo.mae(test_preds)

        # # add a rough time measurement
        # t_1 = time.time()
        # elapsed = t_1 - t_0
        # results['time'] = elapsed

        # all_results.append(results)


        # # Hybrid model, run KNN w/ MF predictions as input

        t_0 = time.time()
        print "Running Hybrid with k of {}".format(k)

        mf_df = pd.DataFrame(mf_preds)
        mf_df = mf_df[['uid', 'iid', 'est']]
        mf_df.columns = ['UserId', 'MovieId', 'Rating']

        # filter the test set
        keys = ['UserId', 'MovieId']
        i1 = test.set_index(keys).index
        i2 = mf_df.set_index(keys).index
        test = test[i1.isin(i2)]

        print len(test)

        # train on all the predictions
        algo = F.custom_knn(mf_df.copy(), mf_df.copy(), similarity = 'adjusted_cosine', by = 'item', k = k)

        train_preds = algo.predict()
        test_preds = algo.predict(test.copy())

        results = {}
        results['sample'] = sample
        results['name'] = 'hybrid'

        # add k, and sample size to results
        results['sample'] = sample
        results['k'] = k

        results['test'] = { 'mae': 0, 'cc': 0.75 }
        results['train'] = { 'mae': 0 , 'cc': 0.75 }

        results['train']['mae'] = algo.mae(train_preds)
        results['test']['mae'] = algo.mae(test_preds)

        # add a rough time measurement
        t_1 = time.time()
        elapsed = t_1 - t_0
        results['time'] = elapsed

        all_results.append(results)


_print = pp.PrettyPrinter(depth = 2, indent = 2)
for results in all_results:
    _print.pprint(results)

_json = json.dumps(all_results)
f = open('../out/mf_hybrid.json', 'w')
f.write(_json)
f.close()
