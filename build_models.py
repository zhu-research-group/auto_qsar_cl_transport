import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

from classic_ml import split_train_test, CLASSIFIER_ALGORITHMS
from config import directory_check
from molecules_and_features import make_dataset
from stats import get_class_stats, class_scoring

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Build QSAR Models')
parser.add_argument('-ds', '--dataset', metavar='ds', type=str, help='training set name')
parser.add_argument('-f', '--features', metavar='f', type=str, help='features to build model with')
parser.add_argument('-ns', '--n_splits', metavar='ns', type=int, help='number of splits for cross validation')
parser.add_argument('-ev', '--env_var', metavar='ev', type=str, help='environmental variable of project directory')
parser.add_argument('-nc', '--name_col', metavar='nc', type=str, help='name of name column in sdf file')
parser.add_argument('-ep', '--endpoint', metavar='ep', type=str, help='endpoint to model')
parser.add_argument('-t', '--threshold', metavar='t', type=int, help='threshold cutoff')
parser.add_argument('-ts', '--test_set_size', metavar='ts', type=float, help='size of the test set')

args = parser.parse_args()

dataset = args.dataset
features = args.features
n_splits = args.n_splits
seed = 0
env_var = args.env_var
data_dir = os.getenv(env_var)
name_col = args.name_col
endpoint = args.endpoint
threshold = args.threshold
test_set_size = args.test_set_size

# Check to see if necessary directories are present and if not, create them
directory_check(data_dir)

# get training data and split in training, test
# and use a seed for reproducibility
X, y = make_dataset(f'{dataset}.sdf', data_dir=env_var, features=features, name_col=name_col, endpoint=endpoint,
                    threshold=threshold)
X_train, y_train_class, X_test, y_test_class = split_train_test(X, y, n_splits, test_set_size, seed, None)

cv = model_selection.StratifiedKFold(shuffle=True, n_splits=n_splits, random_state=seed)

for name, clf, params in CLASSIFIER_ALGORITHMS:
    pipe = pipeline.Pipeline([('scaler', StandardScaler()), (name, clf)])
    grid_search = model_selection.GridSearchCV(pipe, param_grid=params, cv=cv, scoring=class_scoring, refit='AUC')
    grid_search.fit(X_train.values, y_train_class)
    best_estimator = grid_search.best_estimator_

    if name in ['svc']:
        use_name = 'svm'
    
    else:
        use_name = name 

    print(f'\n=======Results for {use_name}=======')

    # save the hyperparameters for the best performing model to csv
    best = grid_search.best_params_
    pd.Series(best).to_csv(os.path.join(
        data_dir, 'results', f'{use_name}_{dataset}_{features}_{endpoint}_{threshold}_best_params.csv'),
        header=False)    

    # get the predictions from the best performing model in 5 fold cv
    cv_predictions = pd.DataFrame(
        cross_val_predict(best_estimator, X_train, y_train_class, cv=cv, method='predict_proba'),
        index=y_train_class.index)
    cv_class = cv_predictions[1].copy()
    cv_class[cv_class >= 0.5] = 1
    cv_class[cv_class < 0.5] = 0
    five_fold_stats = get_class_stats(None, y_train_class, cv_predictions[1])

    # record the predictions and the results
    final_cv_predictions = pd.concat([cv_predictions[1], cv_class], axis=1)
    final_cv_predictions.columns = ['Probability', 'Binary Prediction']
    final_cv_predictions.to_csv(os.path.join(
        data_dir, 'predictions', f'{use_name}_{dataset}_{features}_{endpoint}_{threshold}_{n_splits}fcv_predictions.csv'))

    # print the 5-fold cv accuracy and manually calculated accuracy to ensure they're correct
    print(f'\n{n_splits}-fold cross-validation results:')

    for score, val in five_fold_stats.items():
        print(score, val)

    # write 5-fold cv results to csv
    pd.Series(five_fold_stats).to_csv(os.path.join(
        data_dir, 'results', f'{use_name}_{dataset}_{features}_{endpoint}_{threshold}_{n_splits}fcv_results.csv'),
        header=False)

    # make predictions on training data, then test data
    train_predictions = pd.Series(best_estimator.predict(X_train.values), index=y_train_class.index)
    train_probabilities = pd.Series(best_estimator.predict_proba(X_train.values)[:, 1], index=y_train_class.index)

    print('\nTraining data results:')

    train_stats = get_class_stats(best_estimator, X_train, y_train_class)

    for score, val in train_stats.items():
        print(score, val)  # write training predictions and stats also

    # write it all to files for later
    final_train_predictions = pd.concat([train_probabilities, train_predictions], axis=1)
    final_train_predictions.to_csv(os.path.join(
        data_dir, 'predictions', f'{use_name}_{dataset}_{features}_{endpoint}_{threshold}_train_predictions.csv'))

    if test_set_size not in [0, None]:
        test_predictions = pd.Series(best_estimator.predict(X_test.values), index=y_test_class.index)
        test_probabilities = pd.Series(best_estimator.predict_proba(X_test.values)[:, 1], index=y_test_class.index)
        final_test_predictions = pd.concat([test_probabilities, test_predictions], axis=1)
        final_test_predictions.to_csv(os.path.join(
            data_dir, 'predictions', f'{use_name}_{dataset}_{features}_{endpoint}_{threshold}_test_predictions.csv'))

        print('\nTest data results:')

        test_stats = get_class_stats(best_estimator, X_test, y_test_class)

        for score, val in test_stats.items():
            print(score, val)

    else:
        test_predictions = None
        test_stats = {}

        for score, val in train_stats.items():
            test_stats[score] = np.nan

    pd.DataFrame([train_stats, test_stats], index=['Training', 'Test']).to_csv(
        os.path.join(data_dir, 'results', f'{use_name}_{dataset}_{features}_{endpoint}_{threshold}_train_test_results.csv'))

    # save model
    save_dir = os.path.join(data_dir, 'models', f'{use_name}_{dataset}_{features}_{endpoint}_{threshold}_pipeline.pkl')
    joblib.dump(best_estimator, save_dir)

print(f'{dataset} {features} - success')
