from sklearn import svm, metrics, linear_model, neighbors, tree, discriminant_analysis
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import numpy as np

classifiers = [
    linear_model.LogisticRegression(),
    svm.SVC(),
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),
    neighbors.KNeighborsClassifier(),
    tree.DecisionTreeClassifier()
]


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
        y_pred = e.predict(x_test)
        scores_cross_validation = cross_val_score(e, x_values, y_values, cv=10)
        precision = metrics.precision_score(y_test, y_pred)
        recall = metrics.recall_score(y_test, y_pred)
        score = 2 * (precision * recall) / (precision + recall)

        print("\n*****************\n", e)
        print("Confusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred))
        print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
        print("Score without cross-validation:", score)
        print("Scores with cross-validation k-fold k=10", cross_val_score(e, x_train, y_train, cv=10))
        print("Mean score :", np.mean(scores_cross_validation))
