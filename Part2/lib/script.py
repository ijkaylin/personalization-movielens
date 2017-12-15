import distance as dist
import funcs as F
import numpy as np
import pandas as pd


toy = pd.read_csv('../toy.dat', sep = '::', names = ['UserId', 'MovieId', 'Rating', 'Timestamp'], engine = 'python')
# _100k = pd.read_csv('../ratings100k.dat', sep = '::', names = ['UserId', 'MovieId', 'Rating', 'Timestamp'], engine = 'python')
# ratings = pd.read_csv('../ratings.dat', sep = '::', names = ['UserId', 'MovieId', 'Rating', 'Timestamp'], engine = 'python')

# read in our similarity matrix
# user_counts = ratings['UserId'].value_counts()
# movie_counts = ratings['MovieId'].value_counts()
# subset = F.sample(ratings, user_counts, movie_counts, 30000, 1000)

# write subset to file
# subset.to_csv('../subset.csv')

# read in the subset
# subset = pd.read_csv('../subset.csv', names = ['UserId', 'MovieId', 'Rating'], engine = 'python')

# rm = F.build_user_item_matrix(subset)
# rm = pd.DataFrame.as_matrix(rm)

# scores = np.load('../similarity.txt.npy')

# print 'Finished loading in scores'

# if not scores:
#     raise "No sim matrix"
print 'starting predictions'


rm = F.build_user_item_matrix(toy)
rm = pd.DataFrame.as_matrix(rm)
scores = dist.build_similarity_matrix(toy, by = 'item', similarity = 'cosine')
print "Calling predict"
print rm.shape
print rm.transpose().shape
preds = dist.predict(rm, scores, by = 'user', k = 5)

