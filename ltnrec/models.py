import ltn
import torch
import numpy as np
from ltnrec.metrics import compute_metric, check_metrics


class MatrixFactorization(torch.nn.Module):
    """
    Matrix factorization model.

    The model has inside two matrices: one containing the embeddings of the users of the system, one containing the
    embeddings of the items of the system.
    """
    def __init__(self, n_users, n_items, n_factors, biased=False):
        """
        Construction of the matrix factorization model.

        :param n_users: number of users in the dataset
        :param n_items: number of items in the dataset
        :param n_factors: size of embeddings for users and items
        :param biased: whether the MF model must include user and item biases or not, default to False
        """
        super(MatrixFactorization, self).__init__()
        self.u_emb = torch.nn.Embedding(n_users, n_factors)
        self.i_emb = torch.nn.Embedding(n_items, n_factors)
        self.biased = biased
        if biased:
            self.u_bias = torch.nn.Embedding(n_users, 1)
            self.i_bias = torch.nn.Embedding(n_items, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, u_idx, i_idx, dim=1, normalize=False):
        """
        It computes the scores for the given user-item pairs using the matrix factorization approach (dot product).

        :param u_idx: users for which the score has to be computed
        :param i_idx: items for which the score has to be computed
        :param dim: dimension along which the dot product has to be computed
        :param normalize: whether the output must be normalized in [0., 1.] (e.g., for a predicate) or not (logits)
        :return: predicted scores for given user-item pairs
        """
        pred = torch.sum(self.u_emb(u_idx) * self.i_emb(i_idx), dim=dim, keepdim=True)
        if self.biased:
            pred += self.u_bias(u_idx) + self.i_bias(i_idx)
        return torch.sigmoid(pred.squeeze()) if normalize else pred.squeeze()


class Trainer:
    """
    Abstract base class that any trainer must inherit from.
    """
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def train(self, train_loader, val_loader, val_metric, n_epochs=200, early=None, verbose=10, save_path=None):
        """
        Method for the train of the model.

        :param train_loader: data loader for training data
        :param val_loader: data loader for validation data
        :param val_metric: validation metric name
        :param n_epochs: number of epochs of training, default to 200
        :param early: threshold for early stopping, default to None
        :param verbose: number of epochs to wait for printing training details (every 'verbose' epochs)
        :param save_path: path where to save the best model, default to None
        """
        best_val_score = 0.0
        early_counter = 0
        check_metrics(val_metric)

        for epoch in range(n_epochs):
            # training step
            train_loss = self.train_epoch(train_loader)
            # validation step
            val_score = self.validate(val_loader, val_metric)
            # print epoch data
            if (epoch + 1) % verbose == 0:
                print("Epoch %d - Train loss %.3f - Validation %s %.3f"
                      % (epoch + 1, train_loss, val_metric, val_score))
            # save best model and update early stop counter, if necessary
            if val_score > best_val_score:
                best_val_score = val_score
                early_counter = 0
                if save_path:
                    self.save_model(save_path)
            else:
                early_counter += 1
                if early is not None and early_counter > early:
                    print("Training interrupted due to early stopping")
                    break

    def train_epoch(self, train_loader):
        """
        Method for the training of one single epoch.

        :param train_loader: data loader for training data
        :return: training loss value averaged across training batches
        """
        raise NotImplementedError()

    def predict(self, x, *args, **kwargs):
        """
        Method for performing a prediction of the model.

        :param x: input for which the prediction has to be performed
        :param args: these are the potential additional parameters useful to the model for performing the prediction
        :param kwargs: these are the potential additional parameters useful to the model for performing the prediction
        :return: prediction of the model for the given input
        """

    def validate(self, val_loader, val_metric):
        """
        Method for validating the model.

        :param val_loader: data loader for validation data
        :param val_metric: validation metric name
        :return: validation score based on the given metric averaged across validation batches
        """
        raise NotImplementedError()

    def save_model(self, path):
        """
        Method for saving the model.

        :param path: path where to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)

    def load_model(self, path):
        """
        Method for loading the model.

        :param path: path from which the model has to be loaded.
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def test(self, test_loader, metrics):
        """
        Method for performing the test of the model based on the given test data and test metrics.

        :param test_loader: data loader for test data
        :param metrics: metric name or list of metrics' names that have to be computed
        :return: a dictionary containing the value of each metric average across the test batches
        """
        raise NotImplementedError()


class MFTrainer(Trainer):
    """
    Trainer for the Matrix Factorization model.

    The objective of the Matrix Factorization is to minimize the MSE between the predictions of the model and the
    ground truth.
    """
    def __init__(self, mf_model, optimizer):
        """
        Constructor of the trainer for the MF model.

        :param mf_model: Matrix Factorization model
        :param optimizer: optimizer used for the training of the model
        """
        super(MFTrainer, self).__init__(mf_model, optimizer)
        self.mse = torch.nn.MSELoss()

    def train_epoch(self, train_loader):
        train_loss = 0.0
        for batch_idx, (u_i_pairs, ratings) in enumerate(train_loader):
            self.optimizer.zero_grad()
            loss = self.mse(self.model(u_i_pairs[:, 0], u_i_pairs[:, 1]), ratings)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        return train_loss / len(train_loader)

    def predict(self, x, dim=1, normalize=False):
        """
        Method for performing a prediction of the model.

        :param x: tensor containing the user-item pair for which the prediction has to be computed. The first position
        is the user index, while the second position is the item index
        :param dim: dimension across which the dot product of the MF has to be computed
        :param normalize: whether to normalize the prediction in the range [0., 1.]
        :return: the prediction of the model for the given user-item pair
        """
        u_idx, i_idx = x[:, 0], x[:, 1]
        with torch.no_grad():
            return self.model(u_idx, i_idx, dim, normalize)

    def validate(self, val_loader, val_metric):
        val_score = []
        for batch_idx, (data, ground_truth) in enumerate(val_loader):
            predicted_scores = self.predict(data.view(-1, 2))
            val_score.append(compute_metric(val_metric, predicted_scores.view(ground_truth.shape).numpy(),
                                            ground_truth))
        return np.mean(np.concatenate(val_score))

    def test(self, test_loader, metrics):
        check_metrics(metrics)
        if isinstance(metrics, str):
            metrics = [metrics]

        results = {m: [] for m in metrics}
        for batch_idx, (data, ground_truth) in enumerate(test_loader):
            for m in results:
                predicted_scores = self.predict(data.view(-1, 2))
                results[m].append(compute_metric(m, predicted_scores.view(ground_truth.shape).numpy(), ground_truth))
        for m in results:
            results[m] = np.mean(np.concatenate(results[m]))
        return results


class LTNTrainerMF(MFTrainer):
    """
    Trainer for the Logic Tensor Network with Matrix Factorization as the predictive model for the Likes function.

    The Likes function takes as input a user-item pair and produce an un-normalized score (MF). Ideally, this score
    should be near 1 if the user likes the item, and near 0 if the user dislikes the item.

    The closeness between the predictions of the Likes function and the ground truth provided by the dataset for the
    training user-item pairs is obtained by maximizing the truth value of the predicate Sim. The predicate Sim takes
    as input a predicted score and the ground truth, and returns a value in the range [0., 1.]. Higher the value,
    higher the closeness between the predicted score and the ground truth.
    """
    def __init__(self, mf_model, optimizer):
        """
        Constructor of the trainer for the LTN with MF as base model.

        :param mf_model: Matrix Factorization model to implement the Likes function
        :param optimizer: optimizer used for the training of the model
        """
        super(LTNTrainerMF, self).__init__(mf_model, optimizer)
        self.Likes = ltn.Function(self.model)
        self.Sim = ltn.Predicate(func=lambda pred, gt: torch.exp(-0.2 * torch.square(pred - gt)))
        self.Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
        self.Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier='f')
        self.sat_agg = ltn.fuzzy_ops.SatAgg()

    def train_epoch(self, train_loader):
            train_loss = 0.0
            for batch_idx, (u, i, r) in enumerate(train_loader):
                self.optimizer.zero_grad()
                train_sat = self.Forall(ltn.diag(u, i, r), self.Sim(self.Likes(u, i), r)).value
                loss = 1. - train_sat
                loss.backward()
                self.optimizer.step()
                train_loss += train_sat.item()

            return train_loss / len(train_loader)
