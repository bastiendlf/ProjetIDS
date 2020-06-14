import numpy as np
import pandas as pd
from scipy.stats import zscore

import matplotlib.pyplot as plt

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
    # plt.show()

print("Old data frame length:", len(data))
print("New data frame length:", len(df))
print("Number of rows deleted: ", (len(data) - len(df)))
print("We removed ", ((len(data) - len(df)) / len(data)) * 100, "% of total values amount.")

test = pd.plotting.scatter_matrix(df, alpha=0.2, diagonal='hist')
# plt.show()

correlation = df.corr(method='pearson')
# print(correlation)

# repartition des données dans chaque classe : bon ou mauvais
print("")
good_class = df.loc[df['quality'] == 1].shape[0]  # nombre de vins marqués bons
bad_class = df.loc[df['quality'] == -1].shape[0]  # nombre de vins marqués mauvais
print("Proportion/repartition des données filtrées : \n", good_class / len(df) * 100,"% <- proportion de bons vins\n",
       bad_class / len(df) * 100, "% <- proportion de mauvais vins\n")

print("Hello IDS project")
