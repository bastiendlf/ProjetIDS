import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from scipy.stats import zscore
from sklearn import svm, linear_model, neighbors, tree, discriminant_analysis, metrics
from sklearn.impute import SimpleImputer
from perceptron import Perceptron

columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
           'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']

classifiers = [
    linear_model.LogisticRegression(max_iter=10000),
    svm.SVC(),
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),
    neighbors.KNeighborsClassifier(),
    tree.DecisionTreeClassifier(max_depth=1000),
    Perceptron(nb_x_column=len(columns) - 1, learning_rate=0.08, threshold=100, acceptable_error=220)
]


def manual_cross_validation(x_values: pd.DataFrame, y_values: pd.core.series.Series):
    """
    We manually implemented the cross_val_score function from sklearn to compare results and test perceptron
    :param x_values: pd.DataFrame with input values
    :param y_values: pd.core.series.Series with labels (+1 or -1)
    :return: dictionary containing results
    """
    global classifiers
    results = dict()

    folds = StratifiedKFold(n_splits=10)
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

        results[e.__str__()] = np.average(np.mean(scores))

    return results


def eval_all_classifiers(x_values: pd.DataFrame, y_values: pd.core.series.Series):
    """
    Evaluates all classifier scores with different metrics.
    :param x_values: pd.DataFrame with input values
    :param y_values: pd.core.series.Series with labels (+1 or -1)
    :return: dictionary containing results
    """
    global classifiers

    results = dict()

    x_train, x_test, y_train, y_test = train_test_split(x_values, y_values, test_size=0.25, random_state=0)

    for e in classifiers:
        if e.__str__() != "Perceptron()":
            e.fit(x_train, y_train)
            y_predicted = e.predict(x_test)
            scores_cross_validation = cross_val_score(e, x_values, y_values, cv=10)

            print("\n*****************\n", e)
            print("Confusion Matrix:\n", metrics.confusion_matrix(y_test, y_predicted))
            print("Accuracy:", metrics.accuracy_score(y_test, y_predicted))
            print("Score without cross-validation:", metrics.f1_score(y_test, y_predicted))
            print("Scores with cross-validation k-fold k=10", scores_cross_validation)
            print("Mean score :", np.average(scores_cross_validation))
            results[e.__str__()] = np.average(scores_cross_validation)

    return results


def split_train_validation_test_values(x_values: pd.DataFrame, y_values: pd.core.series.Series):
    """
    Split dataframe into 3 parts : train 60%, data validation (for learning rate) 20% and 20% test
    :param x_values:  pd.DataFrame x values
    :param y_values: pd.core.series.Series labels
    :return: x_train, y_train, x_val, y_val, x_test, y_test
    """
    x_train, x_test, y_train, y_test = train_test_split(x_values, y_values, test_size=0.2,
                                                        random_state=0)  # train = 80%, test = 20%
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25,
                                                      random_state=1)  # train = 60%, val = 20%, test = 20%

    return x_train, y_train, x_val, y_val, x_test, y_test


def eval_perceptron(x_train: pd.DataFrame, y_train: pd.core.series.Series, x_test: pd.DataFrame,
                    y_test: pd.core.series.Series, learning_rate):
    """
    Trains a perceptron classifier and returns the score obtained with the given learning rate.
    :param x_train: pd.DataFrame that contains x values to train perceptron
    :param y_train: pd.core.series.Series that contains labels to train perceptron
    :param x_test: pd.DataFrame that contains x values to test perceptron
    :param y_test: pd.core.series.Series that contains labels to test perceptron
    :param learning_rate: learning rate you want to test
    :return: score
    """

    perceptron = Perceptron(nb_x_column=x_train.shape[1], learning_rate=learning_rate, threshold=500,
                            acceptable_error=200)
    perceptron.fit(x_train, y_train)
    return perceptron.score(x_test=x_test, y_test=y_test)


def eval_learning_rate(x_train: pd.DataFrame, y_train: pd.core.series.Series,
                       x_val: pd.DataFrame, y_val: pd.core.series.Series):
    """
    Trains different perceptron with the given data with different learning rate values.
    :param x_train: pd.DataFrame that contains x values to train perceptron
    :param y_train: pd.core.series.Series that contains labels to train perceptron
    :param x_val: pd.DataFrame that contains x values to determine which value of learning rate is the best
    :param y_val: pd.core.series.Series that contains labels to determine which value of learning rate is the best
    :return: learning rate that make the perceptron get the best score
    """

    scores = list()
    learning_rates = np.linspace(start=0.00001, stop=0.1, num=20)

    for learning_rate in learning_rates:
        scores.append(eval_perceptron(x_train, y_train, x_val, y_val, learning_rate))

    plt.scatter(learning_rates, scores)
    plt.plot(learning_rates, scores)
    plt.title("Evolution of the score according to the learning rate ")
    plt.xlabel('Learning rate')
    plt.ylabel('Score obtained')
    plt.show()

    return learning_rates[scores.index(max(scores))]


def remove_outliers(data: pd.DataFrame):
    """
    Remove outliers from a dataframe based on NAN or ZSCORE (https://fr.wikipedia.org/wiki/Cote_Z_(statistiques)).
    :param data: pd.DataFrame
    :return: pd.DataFrame without outliers
    """
    print("\n***************Remove outliers***************")
    data = data.dropna()
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
    Replace outliers by column median based NAN or on ZSCORE (https://fr.wikipedia.org/wiki/Cote_Z_(statistiques)).
    :param data: pd.DataFrame
    :return: pd.DataFrame with replaced outliers
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


def plot_dataframe_columns(df: pd.DataFrame, title: str = None):
    """
    Plots each column of the dataframe with box plots.
    :param df: pd.DataFrame
    :param title: str
    :return:
    """

    fig, axes = plt.subplots(2, 6)  # create figure and axes

    for index, element in enumerate(list(df.columns.values)[:-1]):
        df.boxplot(column=element, ax=axes.flatten()[index])

    fig.delaxes(axes[1, 5])

    if title is not None:
        plt.title(title, y=2.2)
    plt.show()


def plot_scatter_matrix(df: pd.DataFrame):
    """
    Plots and returns scatter matrix (https://pandas.pydata.org/docs/reference/api/pandas.plotting.scatter_matrix.html).
    :param df: pd.DataFrame
    :return: numpy.ndarray scatter_matrix
    """
    scatter_matrix = pd.plotting.scatter_matrix(df, alpha=0.2, diagonal='hist')
    plt.show()
    return scatter_matrix


def correlation_table(df: pd.DataFrame):
    """
    Computes and returns correlation matrix of a dataframe.
    :param df: pd.DataFrame
    :return: DataFrame with correlation matrix.
    """
    return df.corr(method='pearson')
