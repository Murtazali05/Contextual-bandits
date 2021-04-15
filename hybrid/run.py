import matplotlib.pyplot as plt

from data.data_preprocessing import get_movie_data, get_filtered_data, get_user_data
from hybrid.ctr import ctr_simulator

if __name__ == '__main__':
    alpha_value = 0.5
    n = 30
    user, user_features = get_user_data()
    movie_features = get_movie_data()
    top_movies_index, top_movies_features, reward_mean, filtered_data = get_filtered_data(n, movie_features)
    simulation_hybrid_alpha_05 = ctr_simulator(K_arms=n,
                                               d=29,
                                               k=29 * 19,
                                               alpha=alpha_value,
                                               epochs=2,
                                               top_movies_index=top_movies_index,
                                               top_movies_features=top_movies_features,
                                               filtered_data=filtered_data,
                                               user_features=user_features,
                                               steps_printout=5000)

    plt.plot(simulation_hybrid_alpha_05["aligned_ctr"])
    plt.axhline(y=reward_mean, color="red")
    plt.title("alpha = " + str(0.5))
