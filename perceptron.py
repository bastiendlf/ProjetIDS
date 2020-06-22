import numpy as np


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
        # self.theta = np.zeros(nb_x_column + 1) # funny when init theta with zeros
        self.threshold = threshold  # max number of iteration
        self.learning_rate = learning_rate  # it represents Êta in the lesson
        self.acceptable_error = acceptable_error  # acceptable number of errors during learning

    def predict(self, x_values):
        """
        Predict class (+1 or -1) based on x_values.

        prediction = sign(Theta * x + Theta0)

        :param self:
        :param x_values: values of the element
        :return: +1, -1 or 0 if point located on separator
        """

        sign_function = np.dot(x_values, self.theta[1:]) + self.theta[0]
        return 1 if sign_function > 0. else -1

    def fit(self, x_train, y_train):
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
                # here we compute the number of elements predicted in the wrong class
                prediction = self.predict(x_values)
                if prediction != y:
                    elt_in_wrong_class += 1

            # print(_, "iteration : errors left ->", elt_in_wrong_class)

            if elt_in_wrong_class < self.acceptable_error:
                # if the algorithm can classify enough elements correctly we can stop the learning
                break

            for x_values, y in zip(x_train.values, y_train):
                prediction = self.predict(x_values)
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

    def __str__(self):
        return "Perceptron()"
