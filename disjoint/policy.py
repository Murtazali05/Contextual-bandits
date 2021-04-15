import numpy as np

from arm import linucb_disjoint_arm


class linucb_disjoint_policy():

    def __init__(self, K_arms, d, alpha):
        self.K_arms = K_arms
        self.linucb_arms = [linucb_disjoint_arm(arm_index=i, d=d, alpha=alpha) for i in range(K_arms)]

    def select_arm(self, x_array):
        # Инициализируем ucb
        highest_ucb = -1

        # Массив индексов ручек, которые имеют максимальный UCB.
        candidate_arms = []

        for arm_index in range(self.K_arms):
            # Вычисляем ucb каждой ручки, используя текущие covariates в момент времени t
            arm_ucb = self.linucb_arms[arm_index].calc_UCB(x_array)

            # Если у текущей ручки ucb выше чем текущий highest_ucb
            if arm_ucb > highest_ucb:
                # Устанавливаем новый максимальный ucb
                highest_ucb = arm_ucb

                # Reset candidate_arms list with new entry based on current arm
                candidate_arms = [arm_index]

            # Если есть ручка с одинаковым ucb как у highest_ucb, добавляем в список candidate_arms
            if arm_ucb == highest_ucb:
                candidate_arms.append(arm_index)

        # Выбираем ручку из candidate_arms рандомно (tie breaker)
        chosen_arm = np.random.choice(candidate_arms)

        return chosen_arm

    def store_arm_index(self, arms_index_array):
        # Arms_features_array is multidimension array of shape (K_arms, 1+arm_dimensions), where 1 is for arm_index

        # Loop through all arms to store the individual arms
        for i in range(self.K_arms):
            self.linucb_arms[i].arm_index = arms_index_array[i]