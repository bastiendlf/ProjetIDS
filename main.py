import numpy as np
import pandas as pd
from scipy.stats import zscore
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

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

# scatter_matrix = pd.plotting.scatter_matrix(df, alpha=0.2, diagonal='hist')
# plt.show()

correlation = df.corr(method='pearson')
# print(correlation)

# distribution of data in each class: good or bad
good_wines = df.loc[df['quality'] == 1].shape[0]
bad_wines = df.loc[df['quality'] == -1].shape[0]
print("\nProportion of filtered data \n", "Good wines:", good_wines / len(df) * 100, "%\n",
      "Bad wines:", bad_wines / len(df) * 100, "%\n")

# center and reduce
center_df = pd.DataFrame(preprocessing.scale(df, with_mean='True', with_std='True'), columns=columns)

# separate data in groups : cross validation 1/5 (20% test and 80% training) : https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
# kf = KFold(n_splits = 5) # dataframe separated in 5 folds
# df_train = pd.DataFrame
# df_test = pd.DataFrame
# for train_index, test_index, in kf.split(center_df) :
#     print("Train :", train_index, "Test :", test_index)

df_train, df_test = train_test_split(center_df, test_size=0.25, random_state=0)



print("Hello IDS project")