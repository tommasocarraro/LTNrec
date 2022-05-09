from ltnrec.data import MindReaderDataset
from ltnrec.loaders import TrainingDataLoader, ValDataLoader
from ltnrec.models import MatrixFactorization, MFTrainer, MFTrainerExp
import torch
import numpy as np
torch.manual_seed(123)
np.random.seed(123)

emb_size = [1]
b_size = [64]
lr = [0.001]
lambda_reg = [0.001]
data = MindReaderDataset("./dataset/mr_ntp_all_entities")
val_loader = ValDataLoader(data.validation, 256)
test_loader = ValDataLoader(data.test, 256)
for e in emb_size:
    model = MatrixFactorization(data.n_users, data.n_movies, e, biased=True)
    for l in lr:
        for la in lambda_reg:
            optimizer = torch.optim.Adam(model.parameters(), lr=l, weight_decay=la)
            for b in b_size:
                train_loader = TrainingDataLoader(data.get_user_movie_ratings(), b)
                print("Emb size %d - Batch size %d - Lr %.5f" % (e, b, l))
                ltn_rec_mf = MFTrainerExp(model, optimizer)
                ltn_rec_mf.train(train_loader,
                                 val_loader,
                                 "hit@10",
                                 n_epochs=200,
                                 early=10,
                                 verbose=1,
                                 save_path="./saved_models/test.pth")

                ltn_rec_mf.load_model("./saved_models/test.pth")
                test_metrics = ltn_rec_mf.test(test_loader, ["hit@10", "ndcg@10"])
                print("Test metrics: " + str(test_metrics))