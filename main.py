import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn import preprocessing

from functions import eval_all_classifiers, plot_values, correlation_table

data = pd.read_csv('red_wines.csv')

columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
           'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']

feature_cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
                'free sulfur dioxide',
                'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

# 2.2. Preparing data

# Removing rows with nan values
data = data.dropna()

# Removing outliers https://kite.com/python/answers/how-to-remove-outliers-from-a-pandas-dataframe-in-python)
z_scores = zscore(data)
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
df = data[filtered_entries]

print("Old data frame length:", len(data))
print("New data frame length:", len(df))
print("Number of rows deleted: ", (len(data) - len(df)))
print("We removed ", ((len(data) - len(df)) / len(data)) * 100, "% of total values amount.")

correlation = correlation_table(df)

# distribution of data in each class: good or bad
good_wines = df.loc[df['quality'] == 1].shape[0]
bad_wines = df.loc[df['quality'] == -1].shape[0]
print("\nProportion of filtered data \n", "Good wines:", good_wines / len(df) * 100, "%\n",
      "Bad wines:", bad_wines / len(df) * 100, "%\n")

# center and reduce
center_df = pd.DataFrame(preprocessing.scale(df, with_mean='True', with_std='True'), columns=columns)

print("***************Center values***************")
eval_all_classifiers(x_values=center_df[feature_cols], y_values=df.quality)
# print("***************Raw values***************")
# eval_all_classifiers(x_values=df[feature_cols], y_values=df.quality)

print("Hello IDS project")
