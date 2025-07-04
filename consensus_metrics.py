import pandas as pd
import glob
import numpy as np
from rdkit.Chem import PandasTools
from sklearn import metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score, accuracy_score
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef, precision_score, recall_score, confusion_matrix
import os


def calc_metrics(pred_series, exp, endpoint='Activity', name_col='NSC#'):
    """
    Calculates evaluation metrics of consensus model from averaged predictions of individual QSAR models

    :param pred_series: a pandas DataFrame columns or Series containing the averaged predictions of the individual QSAR
    models (type: list, Series or DataFrame column)
    :param exp: sdf with the experimental or known activities (e.g. original training or validation test set) (type: str)
    :param endpoint: name of the activity column (type: str)
    :param name_col: name of the column containing the names/identifiers of the chemicals in the dataset (type: str)

    :return: a DataFrame with the evaluation metrics of the consensus model
    """

    df = pd.DataFrame(columns=['Recall', 'Specificity', 'CCR', 'Precision/PPV', 'AUC', 'F1-Score'])

    conditions = [(pred_series < 0.5), pred_series >= 0.5]
    values = [0, 1]
    binary_preds = pd.Series(np.select(conditions, values), index=pred_series.index)
    binary_preds = binary_preds.loc[(binary_preds == 0) | (binary_preds == 1)]

    sdf = PandasTools.LoadSDF(exp)
    sdf[endpoint] = sdf[endpoint].astype(float)
    try:
        sdf[name_col] = sdf[name_col].astype(int)
    except:
        sdf[name_col] = sdf[name_col].astype(str)

    selected = sdf.loc[sdf[name_col].isin(binary_preds.index)]
    selected.set_index(name_col, inplace=True)
    selected['Binary'] = binary_preds

    true_positives = len(selected[(selected[endpoint] == 1) & (selected['Binary'] == 1)])
    false_positives = len(selected[(selected[endpoint] == 0) & (selected['Binary'] == 1)])
    true_negatives = len(selected[(selected[endpoint] == 0) & (selected['Binary'] == 0)])
    false_negatives = len(selected[(selected[endpoint] == 1) & (selected['Binary'] == 0)])

    sensitivity = float(true_positives / (true_positives + false_negatives))
    specificity = float(true_negatives / (true_negatives + false_positives))
    CCR = float((sensitivity + specificity) / 2)
    ppv = float(true_positives / (true_positives + false_positives))
    f1_sc = f1_score(selected[endpoint], selected['Binary'])
    fpr_tr, tpr_tr, thresholds_tr = roc_curve(selected[endpoint], pred_series)
    roc_auc = auc(fpr_tr, tpr_tr)

    df.at[0, 'Recall'] = sensitivity
    df.at[0, 'Specificity'] = specificity
    df.at[0, 'CCR'] = CCR
    df.at[0, 'Precision/PPV'] = ppv
    df.at[0, 'AUC'] = roc_auc
    df.at[0, 'F1-Score'] = f1_sc

    return df.T
