import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import cross_val_score, train_test_split, learning_curve
from scipy.stats import zscore
from sklearn import svm, linear_model, neighbors, tree, discriminant_analysis
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from perceptron import Perceptron

columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
           'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']

classifiers = [
    linear_model.LogisticRegression(),
    svm.SVC(),
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),
    neighbors.KNeighborsClassifier(),
    tree.DecisionTreeClassifier()
]


def manual_cross_validation(x_values, y_values):
    """
    TODO Write documentation
    :param x_values:
    :param y_values:
    :return:
    """
    folds = StratifiedKFold(n_splits=5)
    for e in classifiers:
        scores = list()
        for train_index, test_index in folds.split(x_values, y_values):
            x_train = pd.DataFrame()
            x_test = pd.DataFrame()

            x_train = x_train.append([x_values.iloc[train_index]], ignore_index=True)
            x_test = x_test.append([x_values.iloc[test_index]], ignore_index=True)

            y_train = pd.Series()
            y_test = pd.Series()

            y_test = y_test.append([y_values.iloc[test_index]], ignore_index=True)
            y_train = y_train.append([y_values.iloc[train_index]], ignore_index=True)

            e.fit(x_train, y_train)

            scores.append(e.score(x_test, y_test))
        print(e)
        print(scores)
        print(np.mean(scores))


def eval_all_classifiers(x_values: pd.DataFrame, y_values: pd.core.series.Series):
    """
    TODO Write documentation
    :param x_values:
    :param y_values:
    :return:
    """

    x_train, x_test, y_train, y_test = train_test_split(x_values, y_values, test_size=0.25, random_state=0)

    for e in classifiers:
        e.fit(x_train, y_train)
        y_predicted = e.predict(x_test)

        scores_cross_validation = cross_val_score(e, x_values, y_values, cv=5)

        print("\n*****************\n", e)
        print("Confusion Matrix:\n", metrics.confusion_matrix(y_test, y_predicted))
        print("Accuracy:", metrics.accuracy_score(y_test, y_predicted))
        print("Score without cross-validation:", metrics.f1_score(y_test, y_predicted))
        print("Scores with cross-validation k-fold k=10", scores_cross_validation)
        print("Mean score :", np.average(scores_cross_validation))


def eval_perceptron(x_train: pd.DataFrame, y_train: pd.core.series.Series, x_val: pd.DataFrame,
                    y_val: pd.core.series.Series, learning_rate):
    """
    TODO Write documentation
    :param x_train:
    :param y_train:
    :param x_val:
    :param y_val:
    :param learning_rate:
    :return:
    """

    perceptron = Perceptron(x_train.shape[1], learning_rate=learning_rate, threshold=500)

    perceptron.fit(x_train, y_train)
    y_predicted = list()

    for element in x_val.values:
        y_predicted.append(perceptron.predict(element))
    # print("\n*****************\n", perceptron)
    # print("Confusion Matrix:\n", metrics.confusion_matrix(y_test, y_predicted))
    # print("Accuracy:", metrics.accuracy_score(y_test, y_predicted))
    # print("Score without cross-validation:", metrics.f1_score(y_test, y_predicted))
    return metrics.accuracy_score(y_val, y_predicted)


def eval_learning_rate(x_values: pd.DataFrame, y_values: pd.core.series.Series):
    """
    TODO Write documentation
    :param x_values:
    :param y_values:
    :return:
    """
    x_train, x_test, y_train, y_test = train_test_split(x_values, y_values, test_size=0.2,
                                                        random_state=0)  # train = 80%, test = 20%
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25,
                                                      random_state=1)  # train = 60%, val = 20%, test = 20%
    scores = list()
    learning_rates = list(
        [0.000001, 0.000500, 0.001000, 0.001500, 0.002000, 0.002500, 0.003000, 0.003500, 0.004000, 0.004500, 0.005000])

    for accuracy in learning_rates:  # from 10e-6 to 10e-4 with a step of
        scores.append(eval_perceptron(x_train, y_train, x_val, y_val, accuracy))

    plt.scatter(learning_rates, scores, c='red', marker='o')
    plt.show()


def remove_outliers(data: pd.DataFrame):
    """
    TODO Write documentation
    :param data:
    :return:
    """
    print("\n***************Remove outliers***************")
    data = data.dropna()
    # Removing outliers https://kite.com/python/answers/how-to-remove-outliers-from-a-pandas-dataframe-in-python)
    z_scores = zscore(data)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    df = data[filtered_entries]

    print("Old data frame length:", len(data))
    print("New data frame length:", len(df))
    print("Number of rows deleted: ", (len(data) - len(df)))
    print("We removed ", ((len(data) - len(df)) / len(data)) * 100, "% of total individuals amount.")
    return df


def change_outliers_by_median(data: pd.DataFrame):
    """
    TODO Write documentation
    :param data:
    :return:
    """
    print("\n***************Replace outliers by column median***************")
    changed_values = data.isna().sum().sum()
    imp = SimpleImputer(missing_values=np.nan, strategy='median')
    imp = imp.fit(data)
    data = imp.transform(data)
    data = pd.DataFrame(data, columns=columns)

    z_scores = zscore(data)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores > 3)

    test = sum(sum(filtered_entries))

    changed_values += test

    for index, column in enumerate(data.columns):
        median = data[column].median()
        data[column].loc[filtered_entries[:, index]] = median

    print("Total value amount :", data.shape[0] * data.shape[1])
    print("Outlying values found and replaced by the column median:", changed_values)
    print("We replaced", (100 * changed_values / (data.shape[0] * data.shape[1])), "% of total values amount.")
    return data


def plot_dataframe_columns(df: pd.DataFrame):
    """
    TODO Write documentation
    :param df:
    :return:
    """
    fig, axes = plt.subplots(2, 6)  # create figure and axes

    for index, element in enumerate(list(df.columns.values)[:-1]):
        df.boxplot(column=element, ax=axes.flatten()[index])

    fig.delaxes(axes[1, 5])
    plt.show()


def plot_scatter_matrix(df: pd.DataFrame):
    """
    TODO Write documentation
    :param df:
    :return:
    """
    scatter_matrix = pd.plotting.scatter_matrix(df, alpha=0.2, diagonal='hist')
    plt.show()
    return scatter_matrix


def correlation_table(df: pd.DataFrame):
    """
    TODO Write documentation
    :param df:
    :return:
    """
    return df.corr(method='pearson')
