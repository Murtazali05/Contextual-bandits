import numpy as np


# Класс для linear ucb hybrid arm
class linucb_hybrid_arm():
    # Note that shared features coefficients beta_hat is calculated outside of each arm since
    # it is about shared features across individual arms

    def __init__(self, d, k, alpha):
        # Arm specific A: (d * d) matrix = D_a.T * D_a + I_d.
        # The inverse of A is used in ridge regression
        self.A = np.identity(d)

        # B: матрица размерности (d * k)
        # Равно D_a.T * c_a в ridge regression formulation
        self.B = np.zeros([d, k])

        # b: (d * 1) соответствует вектор ответов.
        # Равно D_a.T * c_a в формулировке ridge regression
        self.b = np.zeros([d, 1])

        # Гиперпараметр Alpha
        self.alpha = alpha

    def init_arm_features(self, arm_index, arm_features_array):
        # Track arm_index
        self.index = arm_index

        # arm_features to be used with x_array using outer product to find individual arm z_features
        self.arm_features = arm_features_array

    def calc_UCB(self, x_array, A_node_inv, beta_hat):
        # beta_hat is the coefficients for Z shared features. (k * 1) vector
        # A_node_inv is (k * k) matrix that is inverse for A_node (shared features)

        # Create arm specific z_array with x_array and self.arm_features
        # z_array elements is based on the combination of user and movie features, which is the outer product of both arrays
        # z_array = Outer product = (19 * 29) or (k by d) matrix
        z_array = np.outer(self.arm_features, x_array).reshape(-1, 1)

        # Find inverse of arm-specific A
        A_inv = np.linalg.inv(self.A)

        # Find theta_arm with beta_hat input
        self.theta = np.dot(A_inv, (self.b - np.dot(self.B, beta_hat)))

        # Стандартное отклонение
        s = np.dot(z_array.T, np.dot(A_node_inv, z_array)) \
            - 2 * np.dot(z_array.T, np.dot(A_node_inv, np.dot(self.B.T, np.dot(A_inv, x_array)))) \
            + np.dot(x_array.T, np.dot(A_inv, x_array)) \
            + np.dot(x_array.T,
                     np.dot(A_inv, np.dot(self.B, np.dot(A_node_inv, np.dot(self.B.T, np.dot(A_inv, x_array))))))

        # UCB
        p = np.dot(z_array.T, beta_hat) + np.dot(x_array.T, self.theta) + self.alpha * np.sqrt(s)

        return p

    def reward_update(self, reward, x_array, z_array):
        # Update A which is (d * d) matrix.
        self.A += np.dot(x_array, x_array.T)

        # Update B which is (d * k) matrix.
        self.B += np.dot(x_array, z_array.T)

        # Update b which is (d * 1) vector
        # reward is scalar
        self.b += reward * x_array
