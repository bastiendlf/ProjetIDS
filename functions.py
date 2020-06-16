from sklearn import metrics
from sklearn.model_selection import cross_val_score, train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import zscore
from sklearn import svm, linear_model, neighbors, tree, discriminant_analysis
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer

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
    folds = StratifiedKFold(n_splits=10)
    for e in classifiers:
        scores = list()
        for train_index, test_index in folds.split(x_values, y_values):
            x_train, x_test, y_train, y_test = x_values[train_index], x_values[test_index], \
                                               y_values[train_index], y_values[test_index]

            e.fit(x_train, y_train)
            list.append(e.score(x_test, y_test))
        print(e)
        print(scores)
        print(np.mean(scores))


def eval_all_classifiers(x_values, y_values):
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


def remove_outliers(data):
    """
    TODO Write documentation
    :param data:
    :return:
    """

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
    return df


def change_outliers_by_median(data):
    """
    TODO Write documentation
    :param data:
    :return:
    """
    imp = SimpleImputer(missing_values=np.nan, strategy='median')
    imp = imp.fit(data)
    data = imp.transform(data)
    data = pd.DataFrame(data, columns=columns)

    z_scores = zscore(data)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores > 3)

    for index, column in enumerate(data.columns):
        median = data[column].median()
        data[column].loc[filtered_entries[:, index]] = median

    return data


def plot_values(df, columns):
    """
    TODO Write documentation
    :param df:
    :param columns:
    :return:
    """
    for e in columns:
        df.boxplot(column=e)
        plt.show()


def plot_scatter_matrix(df):
    """
    TODO Write documentation
    :param df:
    :return:
    """
    scatter_matrix = pd.plotting.scatter_matrix(df, alpha=0.2, diagonal='hist')
    plt.show()
    return scatter_matrix


def correlation_table(df):
    """
    TODO Write documentation
    :param df:
    :return:
    """
    return df.corr(method='pearson')
