import numpy as np


class Perceptron:
    """
    Implements a perceptron
    """

    def __init__(self, nb_input: int, learning_rate: float = 0.001, threshold: int = 100):
        self.theta = np.zeros(nb_input + 1)  # we add 1 line because of theta0
        # We choose to init this matrix with zeros but we could init it with random values
        self.threshold = threshold  # max number of iteration
        self.learning_rate = learning_rate  # it represents Êta in the lesson

    def predict(self, x_values):
        """
        TODO Write documentation
        :param self:
        :param x_values:
        :return:
        """
        # prediction = sign(Theta * x + Theta0)
        sign_function = np.dot(x_values, self.theta[1:]) + self.theta[0]
        return 1.0 if sign_function > 0. else -1.0

    def fit(self, x_train, y_train):
        """
        TODO Write documentation
        :param self:
        :param x_train:
        :param y_train:
        :return:
        """
        for _ in range(self.threshold):
            for x_values, y in zip(x_train.values, y_train):
                prediction = self.predict(x_values)
                # Stochastic Gradient Descent :
                # Theta = Theta0 + Êta * (y - y_hat) * x_i
                self.theta[1:] += self.learning_rate * (y - prediction) * x_values
                # Theta0 = Theta0 + Êta * (y - y_hat)
                self.theta[0] += self.learning_rate * (y - prediction)
