import pandas as pd
from perceptron import Perceptron

perceptron = Perceptron(nb_input=2, learning_rate=0.01, threshold=10)

dict = {'A': [1, 1, 0, 0], 'B': [1, 0, 1, 0]}
list = [1, 0, 0, 0]

x_train = pd.DataFrame(dict)
y_train = pd.Series(list)

perceptron.fit(x_train, y_train)

print('(1 1) :', perceptron.predict(pd.DataFrame({'A': [1], 'B': [1]})))
print('(1 0) :', perceptron.predict(pd.DataFrame({'A': [1], 'B': [0]})))
print('(0 1) :', perceptron.predict(pd.DataFrame({'A': [0], 'B': [1]})))
print('(0 0) :', perceptron.predict(pd.DataFrame({'A': [0], 'B': [0]})))
