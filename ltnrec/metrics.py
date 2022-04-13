import numpy as np
valid_metrics = ['ndcg', 'recall', 'hit']


def hit_at_k(pred_scores, ground_truth, k=10):
    """
    Computes the hit ratio (at k) given the predicted scores and relevance of the items.

    :param pred_scores: score vector in output from the recommender (unsorted ranking)
    :param ground_truth: binary vector with relevance data (1 relevant, 0 not relevant)
    :param k: length of the ranking on which the metric has to be computed
    :return: hit ratio at k position
    """
    k = min(pred_scores.shape[1], k)
    # generate ranking
    rank = np.argsort(-pred_scores, axis=1)
    # get relevance of first k items in the ranking
    rank_relevance = ground_truth[np.arange(pred_scores.shape[0])[:, np.newaxis], rank[:, :k]]
    # sum along axis 1 to count number of relevant items on first k-th positions
    # it is enough to have one relevant item in the first k-th for having a hit ratio of 1
    return rank_relevance.sum(axis=1) > 0


def ndcg_at_k(pred_scores, ground_truth, k=10):
    """
    Computes the NDCG (at k) given the predicted scores and relevance of the items.

    :param pred_scores: score vector in output from the recommender (unsorted ranking)
    :param ground_truth: binary vector with relevance data (1 relevant, 0 not relevant)
    :param k: length of the ranking on which the metric has to be computed
    :return: NDCG at k position
    """
    k = min(pred_scores.shape[1], k)
    # compute DCG
    # generate ranking
    rank = np.argsort(-pred_scores, axis=1)
    # get relevance of first k items in the ranking
    rank_relevance = ground_truth[np.arange(pred_scores.shape[0])[:, np.newaxis], rank[:, :k]]
    log_term = 1. / np.log2(np.arange(2, k + 2))
    # compute metric
    dcg = (rank_relevance * log_term).sum(axis=1)
    # compute IDCG
    # idcg is the ideal ranking, so all the relevant items must be at the top, namely all 1 have to be at the top
    idcg = np.array([(log_term[:min(int(n_pos), k)]).sum() for n_pos in ground_truth.sum(axis=1)])
    return dcg / idcg


def recall_at_k(pred_scores, ground_truth, k=10):
    """
    Computes the recall (at k) given the predicted scores and relevance of the items.

    :param pred_scores: score vector in output from the recommender (unsorted ranking)
    :param ground_truth: binary vector with relevance data (1 relevant, 0 not relevant)
    :param k: length of the ranking on which the metric has to be computed
    :return: recall at k position
    """
    k = min(pred_scores.shape[1], k)
    # generate ranking
    rank = np.argsort(-pred_scores, axis=1)
    # get relevance of first k items in the ranking
    rank_relevance = ground_truth[np.arange(pred_scores.shape[0])[:, np.newaxis], rank[:, :k]]
    # sum along axis 1 to count number of relevant items on first k-th positions
    # divide the number of relevant items in fist k positions by the number of relevant items to get recall
    return rank_relevance.sum(axis=1) / np.minimum(k, ground_truth.sum(axis=1))


def check_metrics(metrics):
    """
    Check if the given list of metrics' names is correct.

    :param metrics: list of str containing the name of some metrics
    """
    if isinstance(metrics, str):
        metrics = [metrics]
    assert all([isinstance(m, str) for m in metrics]), "The metrics must be represented as strings"
    assert all(["@" in m for m in metrics]), "The @ is missing on some of the given metrics"
    assert all([m.split("@")[0] in valid_metrics for m in metrics]), "Some of the given metrics are not valid." \
                                                                     "The accepted metrics are " + str(valid_metrics)
    assert all([m.split("@")[1].isdigit() for m in metrics]), "The k must be an integer"


def compute_metric(metric, pred_scores, ground_truth):
    """
    Compute the given metric on the given predictions and ground truth.

    :param metric: name of the metric that has to be computed
    :param pred_scores: score vector in output from the recommender (unsorted ranking)
    :param ground_truth: binary vector with relevance data (1 relevant, 0 not relevant)
    :return: the value of the given metric for the given predictions and relevance
    """
    m, k = metric.split("@")
    k = int(k)

    if m == "ndcg":
        return ndcg_at_k(pred_scores, ground_truth, k=k)
    if m == "hit":
        return hit_at_k(pred_scores, ground_truth, k=k)
    else:
        return recall_at_k(pred_scores, ground_truth, k=k)
