import numpy as np
import pandas as pd


class Perceptron:

    def __init__(self, nb_input: int, learning_rate: int = 0.001, threshold: int = 100):
        self.theta = np.zero(nb_input + 1)  # we add 1 line because of theta0
        # Notice that we choose to init this matrix with zeros but we could have innit it with random values
        self.threshold = threshold  # max number of iteration
        self.learning_rate = learning_rate  # it represents Êta in the lesson


def predict(self, x_values: pd.DataFrame):
    """
    TODO Write documentation
    :param self:
    :param x_values:
    :return:
    """

    y_prediction = pd.Series()

    for current_individual in x_values:
        sign_function = self.theta[0] + np.dot(current_individual, self.theta[1:])  # computing Theta * x + Theta0

        if sign_function > 0:
            y_prediction.append(1)
        else:
            y_prediction.append(-1)

    return y_prediction


def fit(self, x_train: pd.core.frame.DataFrame, y_train: pd.core.series.Series):
    """
    TODO Write documentation
    :param self:
    :param x_train:
    :param y_train:
    :return:
    """
    for _ in range(self.threshold):
        for x, y in zip(x_train, y_train):
            pass
