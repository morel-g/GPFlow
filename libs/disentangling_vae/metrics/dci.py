# Source https://github.com/cianeastwood/qedr/blob/master/quantify.ipynb

import numpy as np
import scipy
from sklearn.linear_model import Lasso

# from six.moves import range
from sklearn import ensemble

TINY = 1e-12


# def compute_dci2(mus_train, ys_train, mus_test, ys_test):
#     """Computes score based on both training and testing codes and factors."""
#     scores = {}
#     importance_matrix, train_err, test_err = compute_importance_gbt(
#         mus_train, ys_train, mus_test, ys_test
#     )
#     assert importance_matrix.shape[0] == mus_train.shape[0]
#     assert importance_matrix.shape[1] == ys_train.shape[0]
#     scores["informativeness_train"] = train_err
#     scores["informativeness_test"] = test_err
#     scores["disentanglement"] = disentanglement(importance_matrix)
#     scores["completeness"] = completeness(importance_matrix)
#     return scores


# def compute_importance_gbt(x_train, y_train, x_test, y_test):
#     """Compute importance based on gradient boosted trees."""
#     num_factors = y_train.shape[0]
#     num_codes = x_train.shape[0]
#     importance_matrix = np.zeros(
#         shape=[num_codes, num_factors], dtype=np.float64
#     )
#     train_loss = []
#     test_loss = []
#     # Assuming dsprites start at 1
#     for i in range(1, num_factors):
#         model = ensemble.GradientBoostingClassifier()
#         model.fit(x_train.T, y_train[i, :])
#         importance_matrix[:, i] = np.abs(model.feature_importances_)
#         train_loss.append(np.mean(model.predict(x_train.T) == y_train[i, :]))
#         test_loss.append(np.mean(model.predict(x_test.T) == y_test[i, :]))
#     return importance_matrix, np.mean(train_loss), np.mean(test_loss)


# def disentanglement_per_code(importance_matrix):
#     """Compute disentanglement score of each code."""
#     # importance_matrix is of shape [num_codes, num_factors].
#     return 1.0 - scipy.stats.entropy(
#         importance_matrix.T + 1e-11, base=importance_matrix.shape[1]
#     )


# def disentanglement(importance_matrix):
#     """Compute the disentanglement score of the representation."""
#     per_code = disentanglement_per_code(importance_matrix)
#     if importance_matrix.sum() == 0.0:
#         importance_matrix = np.ones_like(importance_matrix)
#     code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()

#     return np.sum(per_code * code_importance)


# def completeness_per_factor(importance_matrix):
#     """Compute completeness of each factor."""
#     # importance_matrix is of shape [num_codes, num_factors].
#     return 1.0 - scipy.stats.entropy(
#         importance_matrix + 1e-11, base=importance_matrix.shape[0]
#     )


# def completeness(importance_matrix):
#     """ "Compute completeness of the representation."""
#     per_factor = completeness_per_factor(importance_matrix)
#     if importance_matrix.sum() == 0.0:
#         importance_matrix = np.ones_like(importance_matrix)
#     factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()
#     return np.sum(per_factor * factor_importance)


def entropic_scores(r):
    """r: relative importances"""
    r = np.abs(r) + TINY
    ps = (r) / (np.sum(r, axis=0))  # 'probabilities'
    hs = [1 - norm_entropy(p) for p in ps.T]
    return hs


def norm_entropy(p):
    """p: probabilities"""
    n = p.shape[0]
    return -p.dot(np.log(p + TINY) / np.log(n + TINY))


def mse(predicted, target):
    """mean square error"""
    predicted = (
        predicted[:, None] if len(predicted.shape) == 1 else predicted
    )  # (n,)->(n,1)
    target = (
        target[:, None] if len(target.shape) == 1 else target
    )  # (n,)->(n,1)
    err = predicted - target
    err = err.T.dot(err) / len(err)
    return err[0, 0]  # value not array


def rmse(predicted, target):
    """root mean square error"""
    return np.sqrt(mse(predicted, target))


def nmse(predicted, target):
    """normalized mean square error"""
    return mse(predicted, target) / np.var(target)


def nrmse(predicted, target):
    """normalized root mean square error"""
    return rmse(predicted, target) / np.std(target)


def print_table_pretty(name, values, factor_label, model_names):
    headers = [factor_label + str(i) for i in range(len(values[0]))]
    headers[-1] = "Avg."
    headers = "\t" + "\t".join(headers)
    print("{0}:\n{1}".format(name, headers))

    for i, values in enumerate(values):
        value = ""
        for v in values:
            value += "{0:.2f}".format(v) + "&\t"
        print("{0}\t{1}".format(model_names[i], value))
    print("")


def compute_dci(x_train, y_train, x_test, y_test, regressor=Lasso(alpha=0.02)):
    num_factors = y_train.shape[1]
    # +1 for average
    train_err = np.zeros((num_factors + 1))
    test_err = np.zeros((num_factors + 1))
    R = []

    for i in range(num_factors):

        # fit model
        model = regressor  # (**params[i][j])
        model.fit(x_train, y_train[:, i])

        # predict
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)

        # calculate errors
        train_err[i] = nrmse(y_train_pred, y_train[:, i])
        test_err[i] = nrmse(y_test_pred, y_test[:, i])

        # extract relative importance of each code variable in predicting z_j
        r = getattr(model, "coef_")[:, None]  # [n_c, 1]
        R.append(np.abs(r))

    R = np.hstack(R)  # columnwise, predictions of each z

    # disentanglement
    disent_scores = entropic_scores(R.T)
    c_rel_importance = np.sum(R, 1) / (
        np.sum(R)
    )  # relative importance of each code variable
    disent_w_avg = np.sum(np.array(disent_scores) * c_rel_importance)
    disent_scores.append(disent_w_avg)

    # completeness
    complete_scores = entropic_scores(R)
    complete_avg = np.mean(complete_scores)
    complete_scores.append(complete_avg)

    # informativeness (append averages)
    train_err[-1] = np.mean(train_err[:-1])
    test_err[-1] = np.mean(test_err[:-1])
    scores = {}
    scores["Disentanglement"] = disent_scores
    scores["Completeness"] = complete_scores
    scores["Informativness train"] = train_err
    scores["Informativness test"] = test_err

    return scores
