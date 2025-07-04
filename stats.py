from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score


def get_class_stats(model, X, y):
    """
    
    :param model: If None, assume X == y_true and y == y_pred, else should be a trained model
    :param X: Data to predict
    :param y: correct classes
    :return: 
    """
    if not model:
        predicted_probas = y
        predicted_classes = y.copy()
        predicted_classes[predicted_classes >= 0.5] = 1
        predicted_classes[predicted_classes < 0.5] = 0
        y = X
    else:
        if 'predict_classes' in dir(model):
            predicted_classes = model.predict_classes(X, verbose=0)[:, 0]
            predicted_probas = model.predict_proba(X, verbose=0)[:, 0]
        else:
            predicted_classes = model.predict(X)
            predicted_probas = model.predict_proba(X)[:, 1]

    f1_sc = f1_score(y, predicted_classes, zero_division=0)

    # Sometimes SVM spits out probabilties with of inf
    # so set them as 1
    from numpy import inf
    predicted_probas[predicted_probas == inf] = 1

    fpr_tr, tpr_tr, thresholds_tr = roc_curve(y, predicted_probas)
    roc_auc = auc(fpr_tr, tpr_tr)
    # test classification results

    precision = precision_score(y, predicted_classes, zero_division=0)
    recall = recall_score(y, predicted_classes)

    # Specificity calculation
    tn, fp, fn, tp = confusion_matrix(y, predicted_classes).ravel()
    specificity = tn / (tn + fp)
    ccr = (recall + specificity) / 2

    return {'F1-Score': f1_sc, 'AUC': roc_auc, 'Precision/PPV': precision,
            'Recall': recall, 'Specificity': specificity, 'CCR': ccr}


# scoring dictionary, just a dictionary containing the evaluation metrics passed through a make_scorer()
# fx, necessary for use in GridSearchCV

class_scoring = {'F1-Score': make_scorer(f1_score), 'AUC': make_scorer(roc_auc_score),
                 'Precision': make_scorer(precision_score), 'Recall': make_scorer(recall_score)}
