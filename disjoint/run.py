import matplotlib.pyplot as plt

from ctr import ctr_disjoint_simulator
from data.data_preprocessing import get_user_data, get_movie_data, get_filtered_data

if __name__ == '__main__':
    alpha_value = 0.5
    n = 30
    user, user_features = get_user_data()
    movie_features = get_movie_data()
    top_movies_index, top_movies_features, filtered_data = get_filtered_data(n, movie_features)
    simulation_hybrid_alpha_05 = ctr_disjoint_simulator(K_arms=n,
                                                        d=29,
                                                        alpha=alpha_value,
                                                        epochs=2,
                                                        user_features=user_features,
                                                        filtered_data=filtered_data,
                                                        top_movies_index=top_movies_index,
                                                        steps_printout=5000)

    # print(filtered_data_original.head())
    filtered_data.reward.hist()
    plt.show()

    # 0.3134639433097124
    reward_mean = filtered_data.reward.mean()
    print(reward_mean)

    plt.plot(simulation_hybrid_alpha_05["aligned_ctr"])
    plt.axhline(y=reward_mean, color="red")
    plt.title("alpha = " + str(0.5))
    plt.show()
