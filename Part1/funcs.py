import pandas as pd 
import numpy as np
import surprise as sp

#Define the format of each of the data files
#Movies; MovieID::Title::Genres
moviescol = ['MovieId', 'Title', 'Genres','Action', 'Adventure',
 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']



_movies = pd.read_csv('./movies.dat', sep ='::', names = moviescol, engine='python')
_ratings = pd.read_csv('./ratings100k.dat', sep = '::', names = ['UserId', 'MovieId', 'Rating', 'Timestamp'], engine = 'python')

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
    return subset





# matrix = funcs.build_user_item_matrix(ratings, movies)
