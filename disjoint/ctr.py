import numpy as np
import pandas as pd

from policy import linucb_disjoint_policy


def ctr_disjoint_simulator(K_arms, d, alpha, epochs, top_movies_index, user_features, filtered_data, steps_printout):
    # Инициализируем policy
    linucb_disjoint_policy_object = linucb_disjoint_policy(K_arms=K_arms, d=d, alpha=alpha)

    # Храним специфические arm_index
    linucb_disjoint_policy_object.store_arm_index(top_movies_index.to_numpy())

    # Instantiate trackers
    aligned_time_steps = 0
    cumulative_rewards = 0
    aligned_ctr = []
    unaligned_ctr = []  # for unaligned time steps

    for epoch_iter in range(epochs):

        print("Epoch: " + str(epoch_iter))

        if epoch_iter == 0:
            # Начинаем с filtered data
            data = filtered_data.copy()
            # Инициализируем unused_data df
            unused_data = pd.DataFrame(columns=["user_id", "movie_id", "rating", "reward"])
        else:

            # Переиспользуем неиспользованные данные
            data = unused_data.copy().reset_index(drop=True)
            # Инициализируем unused_data df
            unused_data = pd.DataFrame(columns=["user_id", "movie_id", "rating", "reward"])

        for i in range(len(data)):

            user_id = data.loc[i, "user_id"]
            movie_id = data.loc[i, "movie_id"]

            # x_array: User features
            data_x_array = np.array(
                user_features.query("user_id == @user_id").drop("user_id", axis=1))  # Размер (1 * 29), d = 29
            data_x_array = data_x_array.reshape(29, 1)

            # Получим rewards
            data_reward = data.loc[i, "reward"]

            if i % steps_printout == 0:
                print("step " + str(i))

            # Выбираем ручку в соответствии с policy's на основе контекста текущего шага
            chosen_arm_index = linucb_disjoint_policy_object.select_arm(data_x_array)

            # Проверяем соответствует ли выбранная ручка(arm_index) с ожидаемой ручкой(data_arm).
            # Отметим, что индекс data_arms варьируется от 1 до 10. В то время как индекс policy arms от 0 до 9.
            if linucb_disjoint_policy_object.linucb_arms[chosen_arm_index].arm_index == movie_id:

                # Используем reward выбранной ручки, чтобы обновить данные
                linucb_disjoint_policy_object.linucb_arms[chosen_arm_index].reward_update(data_reward, data_x_array)

                # Вычисляем CTR
                aligned_time_steps += 1
                cumulative_rewards += data_reward
                aligned_ctr.append(cumulative_rewards / aligned_time_steps)

            else:
                # Переиспользуемые данные
                unused_data = unused_data.append(data.iloc[i])

    return {"aligned_time_steps": aligned_time_steps,
            "cumulative_rewards": cumulative_rewards,
            "aligned_ctr": aligned_ctr,
            "policy": linucb_disjoint_policy_object}
