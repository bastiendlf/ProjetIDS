import pandas as pd
from sklearn import preprocessing
from functions import eval_all_classifiers, plot_values, correlation_table, plot_scatter_matrix, remove_outliers, \
    change_outliers_by_mean, manual_cross_validation

data = pd.read_csv('red_wines.csv')

columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
           'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']

feature_cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
                'free sulfur dioxide',
                'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

# 2.2. Preparing data

df_without_outliers = remove_outliers(data)
df_replace_outliers = change_outliers_by_mean(data)

# correlation = correlation_table(df)

# distribution of data in each class: good or bad
good_wines = data.loc[data['quality'] == 1].shape[0]
bad_wines = data.loc[data['quality'] == -1].shape[0]
print("\nProportion in classes: \n", "Good wines:", good_wines / len(data) * 100, "%\n",
      "Bad wines:", bad_wines / len(data) * 100, "%\n")

# center and reduce
center_df = pd.DataFrame(preprocessing.scale(df_without_outliers, with_mean='True', with_std='True'), columns=columns)

print("***************Center values***************")
eval_all_classifiers(x_values=center_df[feature_cols], y_values=df_without_outliers.quality)

# print("***************Raw values***************")
# eval_all_classifiers(x_values=df_without_outliers[feature_cols], y_values=df_without_outliers.quality)

print("Goodbye IDS project")

manual_cross_validation(x_values=center_df[feature_cols], y_values=df_without_outliers.quality)
