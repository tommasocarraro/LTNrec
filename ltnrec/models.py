import torch


class MatrixFactorization(torch.nn.Module):
    """
    Matrix factorization model.

    The model has inside two matrices: one containing the embeddings of the users of the system, one containing the
    embeddings of the items of the system.
    """
    def __init__(self, n_users, n_items, n_factors):
        """
        Construction of the matrix factorization model.

        :param n_users: number of users in the dataset
        :param n_items: number of items in the dataset
        :param n_factors: size of embeddings
        """
        super(MatrixFactorization, self).__init__()
        self.u_emb = torch.nn.Embedding(n_users, n_factors)
        self.i_emb = torch.nn.Embedding(n_items, n_factors)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, u_idx, i_idx, dim=1, predicate=True):
        """
        It computes the scores for the given user-item pairs using the matrix factorization approach (dot product).

        :param u_idx: users for which the score has to be computed
        :param i_idx: items for which the score has to be computed
        :param dim: dimension along which the dot product has to be computed
        :param predicate: whether the output must be normalized in [0., 1.] (predicate) or not (logits)
        :return: predicted scores for given user-item pairs
        """
        if predicate:
            return self.sigmoid(torch.sum(self.u_emb(u_idx) * self.i_emb(i_idx), dim=dim))
        else:
            return torch.sum(self.u_emb(u_idx) * self.i_emb(i_idx), dim=dim)
