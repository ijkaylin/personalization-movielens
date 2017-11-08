# Run offline model
import time
import pandas as pd
import pprint as pp
import funcs as F
import json

moviescol = ['MovieId', 'Title', 'Genres','Action', 'Adventure',
 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

print "Starting the script"

ratings = pd.read_csv('./ratings.dat', sep = '::', names = ['UserId', 'MovieId', 'Rating', 'Timestamp'], engine = 'python')
movies = pd.read_csv('./movies.dat', sep ='::', names = moviescol, engine='python')


samples = [ [5000, 100], [10000, 200], [15000, 500] ]
# vary_items = [ [10000, i] for i in range(25, 125, 25) ]
# vary_users = [ [i, 100] for i in range(5000, 11000, 1000) ]

k_s = range(5, 60, 5)
factor_sizes = range(5, 60, 5)
top_k = 5
all_results = []

user_value_counts = ratings['UserId'].value_counts()
movie_value_counts = ratings['MovieId'].value_counts()

for sample in vary_items:
    i, j = sample
    _dataset = F.sample(ratings, user_value_counts, movie_value_counts, i, j)
    print "Running Baseline, MF, KNN on the dataset with {} users and {} items".format(i, j)

    base, base_test = F.train_baseline(_dataset)
    base_eval = F.evaluate(base, _dataset, base_test)
    base_eval['name'] = 'baseline'
    base_eval['sample'] = sample

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
        # rough timer, not great at measuring v.fast computations but fine for us
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

_json = json.dumps(all_results)
f = open('./output.json', 'w')
f.write(_json)
f.close()



