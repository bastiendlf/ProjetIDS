import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import zscore

data = pd.read_csv('red_wines.csv')
columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
           'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']

# 2.2. Preparing data

# Removing rows with nan values
data = data.dropna()

# Removing outliers
# (method found here : https://kite.com/python/answers/how-to-remove-outliers-from-a-pandas-dataframe-in-python)
z_scores = zscore(data)
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
df = data[filtered_entries]

for e in columns:
    df.boxplot(column=e)
    plt.show()

print("Old data frame length:", len(data))
print("New data frame length:", len(df))
print("Number of rows deleted: ", (len(data) - len(df)))
print("We removed ", ((len(data) - len(df)) / len(data)), "% of total values amount.")

test = pd.plotting.scatter_matrix(df, alpha=0.2, diagonal='hist')
# plt.show()

print("Hello IDS project")
