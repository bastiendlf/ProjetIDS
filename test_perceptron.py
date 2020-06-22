import pandas as pd
from perceptron import Perceptron
import matplotlib.pyplot as plt
import numpy as np


def plot_values(x, y, evaluation, title: str = "Title", separator=None):
    """
    Plots predicted values with separator
    :param x: x axis values
    :param y: y axis values
    :param evaluation: value class (x, y) = 1 or -1
    :param title: graphic's title
    :param separator: equation of separator
    :return:
    """
    for i in range(len(x)):
        color = "blue" if evaluation[i] == 1 else "red"
        plt.scatter(x[i], y[i], c=color, marker='o')

    if separator is not None:
        x_range = np.linspace(min(x), max(x))
        plt.plot(x_range, separator(x_range), label="Separator")
    plt.title(title)
    plt.legend(loc='best')
    plt.show()


perceptron = Perceptron(nb_x_column=2, learning_rate=0.01, threshold=10, acceptable_error=0)

dict1 = {'A': [1, 1, 0, 0], 'B': [1, 0, 1, 0]}
list1 = [1, -1, -1, -1]

x_train = pd.DataFrame(dict1)
y_train = pd.Series(list1)

perceptron.fit(x_train, y_train)

print("*****Testing perceptron for AND door*****")
print("Input : Output")
print('(1 1) ->', perceptron.predict(pd.DataFrame({'A': [1], 'B': [1]})), ': excepted 1.')
print('(1 0) ->', perceptron.predict(pd.DataFrame({'A': [1], 'B': [0]})), ': excepted -1.')
print('(0 1) ->', perceptron.predict(pd.DataFrame({'A': [0], 'B': [1]})), ': excepted -1.')
print('(0 0) ->', perceptron.predict(pd.DataFrame({'A': [0], 'B': [0]})), ': excepted -1.')
plot_values(dict1["A"], dict1["B"], list1, title="Visualization AND door", separator=perceptron.get_equation_2D())

dict2 = {'A': [1, 1, 0, 0], 'B': [1, 0, 1, 0]}
list2 = [1, 1, 1, -1]

x_train2 = pd.DataFrame(dict2)
y_train2 = pd.Series(list2)

perceptron2 = Perceptron(nb_x_column=2, learning_rate=0.01, threshold=10, acceptable_error=0)

perceptron2.fit(x_train2, y_train2)

print("*****Testing perceptron for OR door*****")
print("Input : Output")
print('(1 1) ->', perceptron2.predict(pd.DataFrame({'A': [1], 'B': [1]})), ': excepted 1.')
print('(1 0) ->', perceptron2.predict(pd.DataFrame({'A': [1], 'B': [0]})), ': excepted 1.')
print('(0 1) ->', perceptron2.predict(pd.DataFrame({'A': [0], 'B': [1]})), ': excepted 1.')
print('(0 0) ->', perceptron2.predict(pd.DataFrame({'A': [0], 'B': [0]})), ': excepted -1.')
plot_values(dict2["A"], dict2["B"], list2, title="Visualization OR door", separator=perceptron2.get_equation_2D())

x1 = [0.1, 0.2, 0.5, 0.75, 0.23, 0.45, 13, 10, 9, 12, 11, 10.5]
x2 = [10, 11, 10.5, 13, 11.5, 12, 0.5, 1, 0.2, 0.3, 1.2, 0.33]

dict3 = {'A': x1, 'B': x2}
list3 = [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1]

x_train3 = pd.DataFrame(dict3)
y_train3 = pd.Series(list3)

perceptron3 = Perceptron(nb_x_column=2, learning_rate=0.01, threshold=10, acceptable_error=0)

perceptron3.fit(x_train3, y_train3)

print("*****Testing perceptron for linear separable values*****")
print('(1 11) ->', perceptron3.predict(pd.DataFrame({'X1': [1], 'X2': [11]})), ': excepted 1.')
print('(2 12) ->', perceptron3.predict(pd.DataFrame({'X1': [2], 'X2': [12]})), ': excepted 1.')
print('(10 0) ->', perceptron3.predict(pd.DataFrame({'X1': [10], 'X2': [0]})), ': excepted -1.')
print('(12 2) ->', perceptron3.predict(pd.DataFrame({'X1': [12], 'X2': [2]})), ': excepted -1.')
plot_values(x1, x2, list3, title="Visualization of linear separable values", separator=perceptron3.get_equation_2D())
