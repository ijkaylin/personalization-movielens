This is the code and notebook for Part1 of the project. We used the
movielens dataset with 10m ratings. These can be found in `ratings.dat`.

The notebook file contains a write up and explanation of our methods. 
We sampled subsets of the data by taking the most popular users and items and training a KNN model and matrix factorization model on the subsets.

All of the functional code is in the file called `funcs.py`. `script.py` reads in the data, and trains and tests the model on
a number of different sample sizes and hyper parameter (k in knn, latent factors in svd), and gives summary results about runtime, RMSEs,
MAEs, and some other data, and writes it to a JSON file.

If you'd like to try running the model with arbitrary parameters, simply open the file and edit the samples array and hyperparameter
options as you like. Run `python script.py` inside the Part1 directory;
the summary data will be written to `Part1/output.json`. Keep in 
mind that some of the training may take a very long time (hours), depending on your sample size. This is the offline phase of our model.

We use some of the data from these offline runs in the notebook. 