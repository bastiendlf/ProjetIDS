import pandas as pd
from sklearn import preprocessing
from functions import eval_all_classifiers, plot_dataframe_columns, correlation_table, plot_scatter_matrix, \
    remove_outliers, change_outliers_by_median, manual_cross_validation, eval_perceptron, eval_learning_rate, \
    split_train_validation_test_values

if __name__ == "__main__":
    data = pd.read_csv('red_wines.csv')

    columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
               'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']

    feature_cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
                    'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

    # 2.2. Preparing data
    df_wines = dict()
    df_wines["remove_outliers"] = remove_outliers(data)
    df_wines["replace_outliers"] = change_outliers_by_median(data)

    plot_dataframe_columns(data, "Box plots by column with raw data")
    plot_dataframe_columns(df_wines["remove_outliers"], "Box plots when we remove the outliers")
    plot_dataframe_columns(df_wines["replace_outliers"], "Box plots when we replace the outliers by the median")

    plot_scatter_matrix(df_wines["remove_outliers"])
    plot_scatter_matrix(df_wines["replace_outliers"])

    correlation1 = correlation_table(df_wines["remove_outliers"])
    correlation2 = correlation_table(df_wines["replace_outliers"])

    # distribution of data in each class: good or bad
    good_wines = data.loc[data['quality'] == 1].shape[0]
    bad_wines = data.loc[data['quality'] == -1].shape[0]
    print("\nProportion in classes: \n", "Good wines:", good_wines / len(data) * 100, "%\n",
          "Bad wines:", bad_wines / len(data) * 100, "%\n")

    # center and reduce
    df_wines["replace_center"] = pd.DataFrame(
        preprocessing.scale(df_wines["replace_outliers"], with_mean='True', with_std='True'), columns=columns)
    df_wines["remove_center"] = pd.DataFrame(
        preprocessing.scale(df_wines["remove_outliers"], with_mean='True', with_std='True'), columns=columns)

    # 2.4. Training classifiers

    cross_validation_mean_scores = dict()

    print("\n***************Training classifiers with centered and reduced values when removing outliers**************")
    cross_validation_mean_scores["remove_center"] = eval_all_classifiers(
        x_values=df_wines["remove_center"][feature_cols],
        y_values=df_wines["remove_outliers"].quality)

    print("\n***************Training classifiers with centered and reduced values when replacing outliers*************")
    cross_validation_mean_scores["replace_center"] = eval_all_classifiers(
        x_values=df_wines["replace_center"][feature_cols],
        y_values=df_wines["replace_outliers"].quality)

    # here we want to compare results with our manual cross validation
    cross_validation_mean_scores["manual cross validation with remove center"] = manual_cross_validation(
        x_values=df_wines["replace_center"][feature_cols],
        y_values=df_wines["replace_outliers"].quality)

    print("\n***************Training classifiers with raw values when removing outliers***************")
    cross_validation_mean_scores["remove_outliers"] = eval_all_classifiers(
        x_values=df_wines["remove_outliers"][feature_cols],
        y_values=df_wines["remove_outliers"].quality)

    print("\n***************Training classifiers with raw values when replacing outliers***************")
    cross_validation_mean_scores["replace_outliers"] = eval_all_classifiers(
        x_values=df_wines["replace_outliers"][feature_cols],
        y_values=df_wines["replace_outliers"].quality)

    # Putting all score results into a Dataframe to analyze them quickly
    df_results_comparison = pd.DataFrame(cross_validation_mean_scores)

    # 3.3. Testing perceptron implementation with project data

    """
     * x_train and y_train will contain data for training the perceptron (60%)
     * x_val and y_val will contain data for finding the best learning rate value (20%)
     * x_test and y_test will contain data for testing the perceptron (20%)
    """

    # Splitting data
    x_train, y_train, x_val, y_val, x_test, y_test = split_train_validation_test_values(
        x_values=df_wines["remove_center"][feature_cols], y_values=df_wines["remove_outliers"].quality)

    # Finding the best learning rate
    best_learning_rate = eval_learning_rate(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)

    # Testing the perceptron classifier on test values
    print("\n***************Training perceptron with centered and reduced values when removing outliers***************")
    test_score = eval_perceptron(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                                 learning_rate=best_learning_rate)
    print("Best learning rate found:", best_learning_rate)
    print("Perceptron score on test values with best learning rate:", test_score)
