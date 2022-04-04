import numpy as np
import bottleneck as bn
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


def compute_metrics(pred_scores, ground_truth, metrics, k_values):
    assert all([m in valid_metrics for m in metrics]), "Wrong metrics have been passed. Accepted metrics are " + \
                                                       str(valid_metrics)
    results = {}
    for m in metrics:
        results[m] = {}
        for k in k_values:
            results[m][k] = hit_at_k(pred_scores, ground_truth, k) if m == "hit" else \
                            (ndcg_at_k(pred_scores, ground_truth, k) if m == "ncdg" else
                             recall_at_k(pred_scores, ground_truth, k))
    return results
