import numpy as np

from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier


def split_train_test(X, y, n_split, test_set_size, split_seed, major_subsample=None):
    """ Splits data into training and test sets"""
    assert X.shape[0] == y.shape[0], f'The lengths of X and y do not match X == {X.shape[0]}, y == {y.shape[0]}.'

    if major_subsample is not None:
        if sum(y) > y.shape[0] / 2:
            major_class = 1

        else:
            major_class = 0

        major_class_index = y[y == major_class].index
        major_class_index_remove = np.random.choice(np.array(major_class_index),
                                                    int(major_class_index.shape[0] * (1. - major_subsample)),
                                                    replace=False)
        X.drop(X.index[major_class_index_remove], inplace=True)
        y.drop(y.index[major_class_index_remove], inplace=True)

    # split X into training sets and test set the size of test_set_
    if test_set_size not in [0, None]:
        batch_size = int(y.shape[0] * (1 - test_set_size) // n_split)  # calculating batch size
        train_size = int(batch_size * n_split)
        test_size = y.shape[0] - train_size

        X_train_tmp, X_test, y_train_class_tmp, y_test_class = model_selection.train_test_split(X, y,
                                                                                                train_size=train_size,
                                                                                                test_size=test_size,
                                                                                                stratify=y,
                                                                                                random_state=split_seed)

    else:
        X_train_tmp = X
        y_train_class_tmp = y
        X_test = None
        y_test_class = None

    cv = model_selection.StratifiedKFold(shuffle=True, n_splits=n_split, random_state=split_seed)
    valid_idx = []  # indexes for new train dataset

    for (_, valid) in cv.split(X_train_tmp, y_train_class_tmp):
        valid_idx += valid.tolist()

    X_train = X_train_tmp.iloc[valid_idx]
    y_train_class = y_train_class_tmp.iloc[valid_idx]

    return X_train, y_train_class, X_test, y_test_class


# Algorithms is a list of tuples where item 1 is the name item 2 is a scikit-learn machine learning classifiers and
# item 3 is the parameters to grid search through

seed = 0

CLASSIFIER_ALGORITHMS = [
    ('rf', RandomForestClassifier(max_depth=10,  # max depth 10 to prevent overfitting
                                  class_weight='balanced',
                                  random_state=seed), {'rf__n_estimators': [5, 10, 25]}),
    ('svc', SVC(probability=True,
                class_weight='balanced',
                random_state=seed), {'svc__kernel': ['rbf'],
                                     'svc__gamma': [1e-2, 1e-3],
                                     'svc__C': [1, 10]}),
    ('xgb', XGBClassifier(), {}),
    ('dnn', MLPClassifier(early_stopping=True, random_state=seed), {'dnn__solver': ['sgd', 'adam'],
                                                                    'dnn__hidden_layer_sizes': [(100, 3), (1000, 3)],
                                                                    'dnn__alpha': [1e-2, 1e-3],
                                                                    'dnn__learning_rate': ['constant', 'adaptive']})
]
