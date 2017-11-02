import pandas as pd 
import numpy as np

#Define the format of each of the data files
#Movies; MovieID::Title::Genres
moviescol = ['MovieId', 'Title', 'Genres','Action', 'Adventure',
 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']



# _movies = pd.read_csv('./movies.dat', sep ='::', names = moviescol, engine='python')
# _ratings = pd.read_csv('./ratings100k.dat', sep = '::', names = ['UserId', 'MovieId', 'Rating', 'Timestamp'], engine = 'python')

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

        

def build_user_item_matrix(ratings, movies):
    """
    Return a USERxITEM matrix with values as the user's value for the movie, null otherwise
    Right now not normalized
    ratings.dat: UserID::MovieID::Rating::Timestamp
    movies.dat: MovieId::Title::Tag1|Tag2|Tag3
    @param ratings Dataframe
    @param movies Dataframe
    @returns matrix numpy matrix with a user's ratings per movie
    """
    matrix = pd.pivot(index = 'UserId', columns = 'MovieId', values = 'Rating').fillna(0)
    return matrix









# matrix = funcs.build_user_item_matrix(ratings, movies)
