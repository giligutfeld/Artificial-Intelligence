# Gili Gutfeld 209284512

import pandas as pd
import numpy as np

'''
The RMSE function calculates the root mean squared error (RMSE) of a given collaborative filtering (CF) model and
test set. The function first converts the test set to a numpy array using a pivot table, and then makes a copy of
the CF model's predicted ratings. The RMSE is calculated using the difference between the test set and the
predicted ratings, and the value is rounded to 5 decimal places. The function also prints the RMSE for the
given CF model and strategy.
'''


def RMSE(test_set, cf):
    test = pd.pivot_table(test_set, index='UserId', columns='ProductId', values='Rating', aggfunc=np.sum).to_numpy()
    prediction_nan = np.copy(cf.pred.to_numpy())
    prediction_nan[np.isnan(test)] = np.nan
    rmse = str(np.sqrt(np.nanmean(np.power((prediction_nan - test), 2))).round(5))
    print("RMSE " + cf.strategy + "-based CF" + ": " + rmse)

    if cf.strategy == "item":
        user_item_mat = cf.user_item_matrix.to_numpy()
        pred_nan = np.copy(np.zeros(user_item_mat.shape) + np.nanmean(user_item_mat, axis=1).reshape(-1, 1))
        pred_nan[np.isnan(test)] = np.nan
        rmse = str(np.sqrt(np.nanmean(np.power((pred_nan - test), 2))).round(5))
        print("mean based (benchmark) " + cf.strategy + ": " + rmse)

'''
The get_metrics function calculates either precision or recall at a given value of k (number of recommended items)
for a CF model and test set. The function first creates a list of recommended items for each user in the test set.
Next, it creates a list of relevant items for each user (i.e. items that were rated 3 or higher in the test set).
The function then calculates the metric of interest (either precision or recall) by comparing the recommended
items to the relevant items for each user. The function also prints the calculated metric value for the CF
model and a benchmark (highest-ranked) approach.
'''


def get_metrics(test_set, cf, k, metric_name):
    recommend_list = []
    for x in cf.pred.index:
        recommend_list.append(set(cf.recommend_items(x, k=k)))

    test = pd.pivot_table(test_set, index='UserId', columns='ProductId', values='Rating', aggfunc=np.sum).to_numpy()
    test[test < 3] = np.nan

    relevant_list = []
    for x in ~np.isnan(test):
        relevant_list.append(set(cf.user_item_matrix.columns[x]))

    recommend_list = np.array(recommend_list)
    relevant_list = np.array(relevant_list)

    count = 0
    num_rec_rel = 0
    divider = k

    for rec, rel in zip(recommend_list, relevant_list):
        if len(rel):
            if metric_name == 'Recall':
                divider = len(rel)
            num_rec_rel += len(rec.intersection(rel)) / divider
            count += 1
    metric = str(np.round((num_rec_rel / count), decimals=5))
    print("user-based CF " + metric_name + '@' + str(k) + ": " + metric)

    top_k = np.argsort(-np.nanmean(cf.user_item_matrix.T.to_numpy(), axis=1))
    items = set()
    for i in range(k):
        items.add(cf.user_item_matrix.T.index[top_k[i]])
    benchmark = [items] * len(cf.user_item_matrix.index)

    count = 0
    num_rec_rel = 0

    for rec, rel in zip(benchmark, relevant_list):
        if len(rel) > 0:
            if metric_name == 'Recall':
                divider = len(rel)
            num_rec_rel += len(rec.intersection(rel)) / divider
            count += 1
    metric = str(np.round((num_rec_rel / count), decimals=5))
    print("highest-ranked (benchmark) " + metric_name + '@' + str(k) + ": " + metric)


def precision_at_k(test_set, cf, k):
    get_metrics(test_set, cf, k, 'Precision')


def recall_at_k(test_set, cf, k):
    get_metrics(test_set, cf, k, 'Recall')
