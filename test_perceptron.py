import pandas as pd
from perceptron import Perceptron
import matplotlib.pyplot as plt


def plot_values(x, y, eval):
    for i in range(len(x)):
        if eval[i] == 1:
            color = 'blue'
        else:
            color = 'red'
        plt.scatter(x[i], y[i], c=color, marker='o')
    plt.title('Data visualization')
    plt.show()


perceptron = Perceptron(nb_input=2, learning_rate=0.01, threshold=10)

dict1 = {'A': [1, 1, 0, 0], 'B': [1, 0, 1, 0]}
list1 = [1, -1, -1, -1]
plot_values(dict1["A"], dict1["B"], list1)

x_train = pd.DataFrame(dict1)
y_train = pd.Series(list1)

perceptron.fit(x_train, y_train)

print("*****Testing perceptron for AND dor*****")
print("Input : Output")
print('(1 1) ->', perceptron.predict(pd.DataFrame({'A': [1], 'B': [1]})), ': excepted 1.')
print('(1 0) ->', perceptron.predict(pd.DataFrame({'A': [1], 'B': [0]})), ': excepted -1.')
print('(0 1) ->', perceptron.predict(pd.DataFrame({'A': [0], 'B': [1]})), ': excepted -1.')
print('(0 0) ->', perceptron.predict(pd.DataFrame({'A': [0], 'B': [0]})), ': excepted -1.')

dict2 = {'A': [1, 1, 0, 0], 'B': [1, 0, 1, 0]}
list2 = [1, 1, 1, -1]
plot_values(dict2["A"], dict2["B"], list2)

x_train2 = pd.DataFrame(dict2)
y_train2 = pd.Series(list2)

perceptron2 = Perceptron(nb_input=2, learning_rate=0.01, threshold=10)

perceptron2.fit(x_train2, y_train2)

print("*****Testing perceptron for OR dor*****")
print("Input : Output")
print('(1 1) ->', perceptron2.predict(pd.DataFrame({'A': [1], 'B': [1]})), ': excepted 1.')
print('(1 0) ->', perceptron2.predict(pd.DataFrame({'A': [1], 'B': [0]})), ': excepted 1.')
print('(0 1) ->', perceptron2.predict(pd.DataFrame({'A': [0], 'B': [1]})), ': excepted 1.')
print('(0 0) ->', perceptron2.predict(pd.DataFrame({'A': [0], 'B': [0]})), ': excepted -1.')

x1 = [0.1, 0.2, 0.5, 0.75, 0.23, 0.45, 13, 10, 9, 12, 11, 10.5]
x2 = [10, 11, 10.5, 13, 11.5, 12, 0.5, 1, 0.2, 0.3, 1.2, 0.33]

dict3 = {'A': x1, 'B': x2}
list3 = [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1]
plot_values(x1, x2, list3)

x_train3 = pd.DataFrame(dict3)
y_train3 = pd.Series(list3)

perceptron3 = Perceptron(nb_input=2, learning_rate=0.01, threshold=10)

perceptron3.fit(x_train3, y_train3)

print("*****Testing perceptron for custom values*****")
print('(1 11) ->', perceptron3.predict(pd.DataFrame({'X1': [1], 'X2': [11]})), ': excepted 1.')
print('(2 12) ->', perceptron3.predict(pd.DataFrame({'X1': [2], 'X2': [12]})), ': excepted 1.')
print('(10 0) ->', perceptron3.predict(pd.DataFrame({'X1': [10], 'X2': [0]})), ': excepted -1.')
print('(12 2) ->', perceptron3.predict(pd.DataFrame({'X1': [12], 'X2': [2]})), ': excepted -1.')
