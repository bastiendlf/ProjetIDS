import pandas as pd
from perceptron import Perceptron

perceptron = Perceptron(nb_input=2, learning_rate=0.001, threshold=100)

dict = {'A': [1, 1, 0, 0], 'B': [1, 0, 1, 0]}
list = [1, 0, 0, 0]

x_train = pd.DataFrame(dict)
y_train = pd.Series(list)

perceptron.fit(x_train, y_train)

print("*****Testing perceptron for AND dor*****")
print("Input : Output")
print('(1 1) :', perceptron.predict(pd.DataFrame({'A': [1], 'B': [1]})))
print('(1 0) :', perceptron.predict(pd.DataFrame({'A': [1], 'B': [0]})))
print('(0 1) :', perceptron.predict(pd.DataFrame({'A': [0], 'B': [1]})))
print('(0 0) :', perceptron.predict(pd.DataFrame({'A': [0], 'B': [0]})))

dict2 = {'A': [1, 1, 0, 0], 'B': [1, 0, 1, 0]}
list2 = [1, 1, 1, 0]

x_train2 = pd.DataFrame(dict2)
y_train2 = pd.Series(list2)

perceptron2 = Perceptron(nb_input=2, learning_rate=0.01, threshold=10)

perceptron2.fit(x_train2, y_train2)

print("*****Testing perceptron for OR dor*****")
print("Input : Output")
print('(1 1) :', perceptron2.predict(pd.DataFrame({'A': [1], 'B': [1]})))
print('(1 0) :', perceptron2.predict(pd.DataFrame({'A': [1], 'B': [0]})))
print('(0 1) :', perceptron2.predict(pd.DataFrame({'A': [0], 'B': [1]})))
print('(0 0) :', perceptron2.predict(pd.DataFrame({'A': [0], 'B': [0]})))
