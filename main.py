import pandas as pd
from sklearn import preprocessing
from functions import eval_all_classifiers, plot_dataframe_columns, correlation_table, plot_scatter_matrix, \
    remove_outliers, \
    change_outliers_by_median, manual_cross_validation, eval_perceptron, eval_learning_rate

data = pd.read_csv('red_wines.csv')

columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
           'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']

feature_cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
                'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

# 2.2. Preparing data
df_dict = dict()
df_dict["remove_outliers"] = remove_outliers(data)
df_dict["replace_outliers"] = change_outliers_by_median(data)

# plot_dataframe_columns(data)
# plot_dataframe_columns(df_dict["remove_outliers"])
# plot_dataframe_columns(df_dict["replace_outliers"])

# plot_scatter_matrix(df_dict["remove_outliers"])
# plot_scatter_matrix(df_dict["replace_outliers"])

# correlation1 = correlation_table(df_dict["remove_outliers"])
# correlation2 = correlation_table(df_dict["replace_outliers"])


# distribution of data in each class: good or bad
good_wines = data.loc[data['quality'] == 1].shape[0]
bad_wines = data.loc[data['quality'] == -1].shape[0]
print("\nProportion in classes: \n", "Good wines:", good_wines / len(data) * 100, "%\n",
      "Bad wines:", bad_wines / len(data) * 100, "%\n")

# center and reduce
df_dict["replace_center"] = pd.DataFrame(
    preprocessing.scale(df_dict["replace_outliers"], with_mean='True', with_std='True'), columns=columns)
df_dict["remove_center"] = pd.DataFrame(
    preprocessing.scale(df_dict["remove_outliers"], with_mean='True', with_std='True'), columns=columns)

# 2.4. Training classifiers
print("***************Training classifiers with centered and reduced values***************")
eval_all_classifiers(x_values=df_dict["remove_center"][feature_cols], y_values=df_dict["remove_outliers"].quality)
# eval_all_classifiers(x_values=df_dict["replace_center"][feature_cols], y_values=df_dict["replace_outliers"].quality)

# print("***************Raw values***************")
# eval_all_classifiers(x_values=df_dict["remove_outliers"][feature_cols], y_values=df_dict["remove_outliers"].quality)

print("Goodbye IDS project")

# manual_cross_validation(x_values=df_dict["remove_center"][feature_cols], y_values=df_dict["remove_outliers"].quality)

# 3.3. Testing perceptron implementation with project data

# eval_perceptron(x_values=df_dict["replace_outliers"][feature_cols], y_values=df_dict["replace_outliers"].quality)
eval_learning_rate(x_values=df_dict["remove_center"][feature_cols], y_values=df_dict["remove_outliers"].quality)
