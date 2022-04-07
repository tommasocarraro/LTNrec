import ltn
import torch
import numpy as np
from ltnrec.metrics import hit_at_k, ndcg_at_k


class MatrixFactorization(torch.nn.Module):
    """
    Matrix factorization model.

    The model has inside two matrices: one containing the embeddings of the users of the system, one containing the
    embeddings of the items of the system.
    """
    def __init__(self, n_users, n_items, n_factors, biased=True):
        """
        Construction of the matrix factorization model.

        :param n_users: number of users in the dataset
        :param n_items: number of items in the dataset
        :param n_factors: size of embeddings
        :param biased: whether the MF model include user and item biases, default to True
        """
        super(MatrixFactorization, self).__init__()
        self.u_emb = torch.nn.Embedding(n_users, n_factors)
        self.i_emb = torch.nn.Embedding(n_items, n_factors)
        self.biased = biased
        if biased:
            self.u_bias = torch.nn.Embedding(n_users, 1)
            self.i_bias = torch.nn.Embedding(n_items, 1)
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
        pred = torch.sum(self.u_emb(u_idx) * self.i_emb(i_idx), dim=dim, keepdim=True)
        if self.biased:
            pred += self.u_bias(u_idx) + self.i_bias(i_idx)
        return self.sigmoid(pred.squeeze()) if predicate else pred.squeeze()


class LTNRecMF:
    def __init__(self, n_users, n_items, emb_size, biased):
        self.model = MatrixFactorization(n_users, n_items, emb_size, biased)
        self.Likes = ltn.Predicate(self.model)
        self.Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
        self.Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier='f')
        self.sat_agg = ltn.fuzzy_ops.SatAgg()

    def train(self, train_loader, val_loader, n_epochs, optimizer, lambda_reg=None, early=None, verbose=10,
              save_path=None):
        best_val = 0.0
        early_counter = 0
        for epoch in range(n_epochs):
            mean_sat = 0.0
            # train step
            for batch_idx, ((u_pos, i_pos), (u_neg, i_neg)) in enumerate(train_loader):
                optimizer.zero_grad()
                f1 = self.Forall(ltn.diag(u_pos, i_pos), self.Likes(u_pos, i_pos))
                f2 = self.Forall(ltn.diag(u_neg, i_neg), self.Not(self.Likes(u_neg, i_neg)))
                train_sat = self.sat_agg(f1, f2)
                loss = (1. - train_sat) + \
                       (lambda_reg * (torch.sum(torch.pow(self.model.u_emb.weight, 2)) +
                                      torch.sum(torch.pow(self.model.i_emb.weight, 2)) +
                                      (torch.sum(torch.pow(self.model.u_bias.weight, 2)) if self.model.biased else 0.0) +
                                      (torch.sum(torch.pow(self.model.i_bias.weight,2)) if self.model.biased else 0.0))
                        if lambda_reg is not None else 0.0)
                loss.backward()
                optimizer.step()
                mean_sat += train_sat.item()
            # validation step
            hit = []
            for batch_idx, (data, ground_truth) in enumerate(val_loader):
                predicted_scores = self.Likes.model(data[:, :, 0], data[:, :, 1], dim=2, predicate=False)
                hit.append(hit_at_k(predicted_scores.detach().numpy(), ground_truth))
            hit = np.mean(np.concatenate(hit))

            if (epoch + 1) % verbose == 0:
                print("Epoch %d - Train SAT %.3f - Validation hit@10 %.3f"
                      % (epoch + 1, mean_sat / len(train_loader), hit))

            if hit > best_val:
                best_val = hit
                early_counter = 0
                if save_path:
                    self.save_model(save_path)
            else:
                early_counter += 1
                if early is not None and early_counter > early:
                    print("Training interrupted due to early stopping")
                    break

    def test(self, test_loader, k):
        hit, ndcg = [], []
        for batch_idx, (data, ground_truth) in enumerate(test_loader):
            predicted_scores = self.model(data[:, :, 0], data[:, :, 1], dim=2, predicate=False)
            hit.append(hit_at_k(predicted_scores.detach().numpy(), ground_truth, k=k))
            ndcg.append(ndcg_at_k(predicted_scores.detach().numpy(), ground_truth, k=k))
        hit = np.mean(np.concatenate(hit))
        ndcg = np.mean(np.concatenate(ndcg))
        return hit, ndcg

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict()
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])


class MF:
    def __init__(self, n_users, n_items, emb_size, biased):
        self.model = MatrixFactorization(n_users, n_items, emb_size, biased)
        self.mse = torch.nn.MSELoss()

    def train(self, train_loader, val_loader, n_epochs, optimizer, lambda_reg=None, early=None, verbose=10,
              save_path=None):
        best_val = 0.0
        early_counter = 0
        for epoch in range(n_epochs):
            mean_mse = 0.0
            # train step
            for batch_idx, (u_i_pairs, ratings) in enumerate(train_loader):
                optimizer.zero_grad()
                loss = self.mse(self.model(u_i_pairs[:, 0], u_i_pairs[:, 1], predicate=False), ratings) + \
                               (lambda_reg * (torch.sum(torch.pow(self.model.u_emb.weight, 2)) +
                                              torch.sum(torch.pow(self.model.i_emb.weight, 2)) +
                                              (torch.sum(torch.pow(self.model.u_bias.weight, 2)) if self.model.biased else 0.0) +
                                              (torch.sum(torch.pow(self.model.i_bias.weight, 2)) if self.model.biased else 0.0))
                                if lambda_reg is not None else 0.0)
                loss.backward()
                optimizer.step()
                mean_mse += loss.item()
            # validation step
            hit = []
            for batch_idx, (data, ground_truth) in enumerate(val_loader):
                predicted_scores = self.model(data[:, :, 0], data[:, :, 1], dim=2, predicate=False)
                hit.append(hit_at_k(predicted_scores.detach().numpy(), ground_truth))
            hit = np.mean(np.concatenate(hit))

            if (epoch + 1) % verbose == 0:
                print("Epoch %d - Train MSE %.3f - Validation hit@10 %.3f"
                      % (epoch + 1, mean_mse / len(train_loader), hit))

            if hit > best_val:
                best_val = hit
                early_counter = 0
                if save_path:
                    self.save_model(save_path)
            else:
                early_counter += 1
                if early is not None and early_counter > early:
                    print("Training interrupted due to early stopping")
                    break

    def test(self, test_loader, k):
        hit, ndcg = [], []
        for batch_idx, (data, ground_truth) in enumerate(test_loader):
            predicted_scores = self.model(data[:, :, 0], data[:, :, 1], dim=2, predicate=False)
            hit.append(hit_at_k(predicted_scores.detach().numpy(), ground_truth, k=k))
            ndcg.append(ndcg_at_k(predicted_scores.detach().numpy(), ground_truth, k=k))
        hit = np.mean(np.concatenate(hit))
        ndcg = np.mean(np.concatenate(ndcg))
        return hit, ndcg

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict()
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
