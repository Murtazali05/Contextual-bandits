import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def get_user_data():
    user = pd.read_csv("../data/ml-100k/u.user", header=None, sep="|")
    user.columns = ["user_id", "age", "gender", "occupation", "zipcode"]
    user = user.drop(["zipcode"], axis=1)

    bins = [0, 20, 30, 40, 50, 60, np.inf]
    names = ['<20', '20-29', '30-39', '40-49', '51-60', '60+']

    user['agegroup'] = pd.cut(user['age'], bins, labels=names)
    user = user.drop(["age"], axis=1)

    columnsToEncode = ["agegroup", "gender", "occupation"]
    myEncoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    myEncoder.fit(user[columnsToEncode])

    user_features = pd.concat([user.drop(columnsToEncode, 1),
                               pd.DataFrame(myEncoder.transform(user[columnsToEncode]),
                                            columns=myEncoder.get_feature_names(columnsToEncode))], axis=1).reindex()

    return user, user_features


def get_movie_data():
    movie = pd.read_csv("../data/ml-100k/u.item", header=None, sep="|", encoding='latin-1')
    movie.columns = ["movie_id", "movie_title", "release_date", "video_release_date", "IMDb_URL",
                     "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary",
                     "Drama",
                     "Fantasy",
                     "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]

    movie_features = movie.drop(["movie_title", "release_date", "video_release_date", "IMDb_URL"], axis=1)
    return movie_features


# Данные пользователей с рейтингами фильмов и reward
def get_filtered_data(n, movie_features):
    data = pd.read_csv("../data/ml-100k/u.data", sep="\t", header=None,
                       names=["user_id", "movie_id", "rating", "timestamp"])
    data = data.drop(["timestamp"], axis=1)

    # Найдем топ фильмы по количеству пользовательских оценок
    top_movies_index = data.groupby("movie_id").count().sort_values("user_id", ascending=False).head(n).reset_index()[
        "movie_id"]

    top_movies_features = movie_features[movie_features.movie_id.isin(top_movies_index)]

    filtered_data_original = data[data["movie_id"].isin(top_movies_index)]
    filtered_data_original["reward"] = np.where(filtered_data_original["rating"] < 5, 0, 1)
    filtered_data_original = filtered_data_original.reset_index(drop=True)

    # Рандомно переставляем строки, чтобы имитировать среду
    np.random.seed(100)
    filtered_data = filtered_data_original.reindex(np.random.permutation(filtered_data_original.index)).reset_index(
        drop=True)

    return top_movies_index, top_movies_features, filtered_data
