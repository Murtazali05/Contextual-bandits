import numpy as np
import pandas as pd

from hybrid.policy import linucb_hybrid_policy


def ctr_simulator(K_arms, d, k, alpha, epochs, top_movies_index, top_movies_features,
                  filtered_data, user_features, steps_printout):
    # Инициализируем policy
    linucb_hybrid_policy_object = linucb_hybrid_policy(K_arms=K_arms, d=d, k=k, alpha=alpha)

    # Store arm specific features
    linucb_hybrid_policy_object.store_arm_features(top_movies_features.to_numpy())

    # Instantiate trackers
    aligned_time_steps = 0
    cumulative_rewards = 0
    aligned_ctr = []
    unaligned_ctr = []  # for unaligned time steps

    for epoch_iter in range(epochs):

        print("Epoch: " + str(epoch_iter))

        if epoch_iter == 0:
            # Start with filtered data first
            data = filtered_data.copy()
            # Initiate unused_data df
            unused_data = pd.DataFrame(columns=["user_id", "movie_id", "rating", "reward"])
        else:

            # Recycle unused data
            data = unused_data.copy().reset_index(drop=True)
            # Initiate unused_data df
            unused_data = pd.DataFrame(columns=["user_id", "movie_id", "rating", "reward"])

        for i in range(len(data)):

            user_id = data.loc[i, "user_id"]
            movie_id = data.loc[i, "movie_id"]

            # x_array: User features
            data_x_array = np.array(
                user_features.query("user_id == @user_id").drop("user_id", axis=1))  # Shape (1 * 29), d = 29
            data_x_array = data_x_array.reshape(29, 1)

            # Obtain rewards
            data_reward = data.loc[i, "reward"]

            if i % steps_printout == 0:
                print("step " + str(i))

            # Find policy's chosen arm based on input covariates at current time step
            chosen_arm_index = linucb_hybrid_policy_object.select_arm(data_x_array)

            # Check if arm_index is the same as data_arm (ie same actions were chosen)
            # Note that data_arms index range from 1 to 10 while policy arms index range from 0 to 9.
            if linucb_hybrid_policy_object.linucb_arms[chosen_arm_index].index == movie_id:

                # Phase 1: Update shared feature matrices A_node, b_node in policy object
                linucb_hybrid_policy_object.update_shared_features_matrices_phase1(
                    linucb_hybrid_policy_object.linucb_arms[chosen_arm_index].B,
                    linucb_hybrid_policy_object.linucb_arms[chosen_arm_index].A,
                    linucb_hybrid_policy_object.linucb_arms[chosen_arm_index].b)

                # Extract chosen_arm arm_features to create z_array
                data_z_array = np.outer(linucb_hybrid_policy_object.linucb_arms[chosen_arm_index].arm_features,
                                        data_x_array).reshape(-1, 1)

                # Use reward information for the chosen arm to update
                linucb_hybrid_policy_object.linucb_arms[chosen_arm_index].reward_update(data_reward, data_x_array,
                                                                                        data_z_array)

                # Phase 2: Update shared feature matrices A_node, b_node in policy object
                linucb_hybrid_policy_object.update_shared_features_matrices_phase2(data_z_array,
                                                                                   data_reward,
                                                                                   linucb_hybrid_policy_object.linucb_arms[
                                                                                       chosen_arm_index].B,
                                                                                   linucb_hybrid_policy_object.linucb_arms[
                                                                                       chosen_arm_index].A,
                                                                                   linucb_hybrid_policy_object.linucb_arms[
                                                                                       chosen_arm_index].b)

                # For CTR calculation
                aligned_time_steps += 1
                cumulative_rewards += data_reward
                aligned_ctr.append(cumulative_rewards / aligned_time_steps)

            else:
                # Recycle data
                unused_data = unused_data.append(data.iloc[i])

    return {"aligned_time_steps": aligned_time_steps,
            "cumulative_rewards": cumulative_rewards,
            "aligned_ctr": aligned_ctr,
            "policy": linucb_hybrid_policy_object}
