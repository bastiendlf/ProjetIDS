import numpy as np
import pandas as pd
from sklearn import metrics


class Perceptron:
    """
    Implements a perceptron
    """

    def __init__(self, nb_x_column: int, learning_rate: float = 0.001, threshold: int = 100, acceptable_error: int = 1):
        """
        Constructor
        :param nb_x_column: number of columns in pandas.Dataframe for x values only
        :param learning_rate: learning_rate (used for Stochastic Gradient Descent)
        :param threshold: threshold to limit a max number of iteration in case if there is no convergence
        :param acceptable_error: number of elements that can be misspredicted during learning
        """
        self.theta = np.random.rand(nb_x_column + 1)  # we add 1 line because of theta0
        # self.theta = np.zeros(nb_x_column + 1) # funny results when init theta with zeros
        self.threshold = threshold  # max number of iteration
        self.learning_rate = learning_rate  # it represents Êta in the lesson
        self.acceptable_error = acceptable_error  # acceptable number of errors during learning

    def predict_one_individual(self, x_value: pd.core.series.Series):
        """
        Predict class (+1 or -1) based on x_value.

        prediction = sign(Theta * x + Theta0)

        :param self:
        :param x_value: values of the element
        :return: +1, -1  (if point located on separator -1)
        """

        sign_function = np.dot(x_value, self.theta[1:]) + self.theta[0]
        return 1 if sign_function > 0. else -1

    def predict(self, x_values: pd.DataFrame):
        """
        Predict class (+1 or -1) based on x_value for multiple individuals.

        prediction = sign(Theta * x + Theta0)
        :param x_values:
        :return: list of predictions
        """

        y_predicted = list()

        for element in x_values.values:
            y_predicted.append(self.predict_one_individual(element))
        return y_predicted

    def fit(self, x_train: pd.DataFrame, y_train: pd.core.series.Series):
        """
        Trains perceptron with x_values and y_values (to adjust self.theta) based on Stochastic Gradient Descent method.
        :param self:
        :param x_train: x_values to train perceptron
        :param y_train: +1 or -1 associated with x_values
        :return:
        """

        for _ in range(self.threshold):
            elt_in_wrong_class = 0

            for x_values, y in zip(x_train.values, y_train):
                # here we compute the number of elements currently predicted in the wrong class
                prediction = self.predict_one_individual(x_values)
                if prediction != y:
                    elt_in_wrong_class += 1

            if elt_in_wrong_class < self.acceptable_error:
                # if the algorithm can classify enough elements correctly we can stop the learning
                break

            for x_values, y in zip(x_train.values, y_train):
                prediction = self.predict_one_individual(x_values)
                # Stochastic Gradient Descent :
                if prediction != y:
                    # Theta(t+1) = Theta(t) + (Êta * y  * x_i)
                    self.theta[1:] += self.learning_rate * y * x_values
                    # Theta0(t+1) = Theta0(t) + (Êta * y)
                    self.theta[0] += self.learning_rate * y

    def get_equation_2D(self):
        """
        Get 2D equation of separator

        y*theta[2] + x*theta[1] + theta[0] = 0
        so we deduce that y = -1 / self.theta[2] * (self.theta[0] + self.theta[1] * x)

        :return: A lambda function that computes the separator 2D equation (y = ax + b)
        """

        return lambda x: -1 / self.theta[2] * (self.theta[0] + self.theta[1] * x)

    def score(self, x_test: pd.DataFrame, y_test: pd.core.series.Series):
        """
        Computes score with the given x and y values
        :param x_test: pd.DataFrame
        :param y_test: pd.core.series.Series
        :return: score obtained by perceptron
        """

        y_predicted = self.predict(x_test)
        return metrics.f1_score(y_test, y_predicted)

    def __str__(self):
        return "Perceptron()"
