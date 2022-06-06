from ltnrec.loaders import TrainingDataLoader, ValDataLoader, TrainingDataLoaderLTN, TrainingDataLoaderLTNGenres
from ltnrec.models import MatrixFactorization, MFTrainer, LTNTrainerMF, LTNTrainerMFGenres, LTNTrainerMFGenres2
import torch
import numpy as np
import json
import os
from multiprocessing import Pool, Manager


def grid_search(model_name, fold, fold_percentage, data, seed=None):
    """
    It performs a grid search of the given model with configured parameters (config/models_config_grid_search.json)
    on the given data.

    :param model_name: str indicating the name of the model on which the grid search has to be computed (i.e., mf, ltn,
    ltn-genres)
    :param fold: train set on which the grid search has to be computed
    :param fold_percentage: percentage of training ratings in fold, just for saving files
    :param data: dataset on which the grid search has to be computed
    :param seed: seed for reproducing the grid search
    """
    if seed:
        # set seed for reproducibility of the grid search
        np.random.seed(seed)
        torch.manual_seed(seed)

    # create loader to validate the model
    val_loader = ValDataLoader(data.validation, 256)

    save_path = "./saved_models/%s-%.1f.pth" % (model_name, fold_percentage)

    if os.path.exists(save_path):
        print("Best weights for %s model on dataset with %.1f of the ratings already exist" % (model_name,
                                                                                               fold_percentage))
    else:
        json_path = save_path.replace(".pth", ".json")

        with open("./config/models_config_grid_search.json") as f:
            config_file = json.load(f)

        print("Starting grid search for %s model on dataset with %.1f of ratings" % (model_name, fold_percentage))
        config = config_file[model_name]
        print("Parameters to be tested: " + str(config))

        best_val = 0.0
        best_params = None

        # begin grid search
        for emb_size in config["emb_size"]:
            for lr in config["lr"]:
                for reg_coeff in config["reg_coeff"]:
                    for b_size in config["batch_size"]:
                        for mf_bias in config["mf_bias"]:
                            for alpha in config["alpha"]:
                                for p in config["p"]:
                                    print("emb size %d - batch size %d - lr %.5f - l2 reg coeff %.5f - "
                                          "mf bias %d - alpha %s - p %s" %
                                          (emb_size, b_size, lr, reg_coeff, mf_bias, str(alpha), str(p)))

                                    model = MatrixFactorization(data.n_users, data.n_movies, emb_size, biased=mf_bias)
                                    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg_coeff)

                                    train_loader = TrainingDataLoader(fold, b_size) if model_name == "mf" else \
                                        (TrainingDataLoaderLTN(fold, b_size) if model_name == "ltn" else
                                         (TrainingDataLoaderLTNGenres(fold,
                                                                      data.get_user_genre_ratings_dict(),
                                                                      data.movie_to_genres,
                                                                      b_size)))

                                    trainer = MFTrainer(model, optimizer) if model_name == "mf" else \
                                        (LTNTrainerMF(model, optimizer, alpha) if model_name == "ltn" else
                                         LTNTrainerMFGenres(model, optimizer, alpha, p))

                                    trainer.train(train_loader,
                                                  val_loader,
                                                  "hit@10",
                                                  n_epochs=200,
                                                  early=20,
                                                  verbose=1,
                                                  save_path=save_path.replace(model_name, model_name + "_"))

                                    trainer.load_model(save_path.replace(model_name, model_name + "_"))
                                    val_score = trainer.test(val_loader, "hit@10")["hit@10"]

                                    if val_score > best_val:
                                        print("New best validation score - hit@10 %.5f" % val_score)
                                        best_val = val_score
                                        trainer.save_model(save_path)
                                        best_params = {
                                            "emb_size": emb_size,
                                            "mf_bias": mf_bias,
                                            "lr": lr,
                                            "reg_coeff": reg_coeff,
                                            "batch_size": b_size,
                                            "alpha": alpha,
                                            "p": p
                                        }

                                        with open(json_path, 'w') as fp:
                                            json.dump(best_params, fp, indent=4)

        print("Best validation score for %s on dataset with %.1f of ratings is hit@10 %.4f" % (model_name,
                                                                                               fold_percentage,
                                                                                               best_val))
        print("The best parameters are %s" % str(best_params))
        # remove temporary file for grid search
        os.remove(save_path.replace(model_name, model_name + "_"))


def experiment(models, data, proportions, test_metrics, report_save_path, starting_seed=0, n_runs=10, parallelize=False):
    """
    For each of the given models:
        1. Loading of model parameters:
            a. if a model configuration does not exist, it runs the grid search with the configuration at the
            config folder
            b. if a model configuration exists, it loads the configuration
        2. Training and testing of model:
            a. for each of the given rating proportions
                - it creates the dataset with the given rating proportion
                - it trains the model on the constructed dataset
                - it test the model on the constructed dataset

    At the end of this procedure, a csv file containing all the information is built.

    :param models: list of models' names on which the experiment has to be performed
    :param data: dataset on which the experiment has to be performed
    :param proportions: proportions of training ratings on which the models have to be trained and tested
    :param test_metrics: list of metrics name that have to be used as test metrics for the test of the models
    :param report_save_path: path where to save a dictionary containing a report of the performed experiments
    :param starting_seed: seed for the first run of the experiment. The seeds for the other runs are simply the
    successors of the starting seed
    :param n_runs: number of times the experiments have to be run. Then, the metrics are averaged across these runs
    :param parallelize: whether the experiments have to be parallelized and distributed across the available processors.
    Note that in case of parallelization, the training logs on the console will be unordered.
    """
    # prepare dictionary that has to be shared across different concurrent runs
    if parallelize:
        manager = Manager()
        results = manager.dict({model: manager.dict({seed: manager.dict({fold: {} for fold in proportions})
                                                     for seed in range(starting_seed, starting_seed + n_runs)})
                                for model in models})
    else:
        results = {model: {seed: {fold: {} for fold in proportions}
                           for seed in range(starting_seed, starting_seed + n_runs)}
                   for model in models}
    
    # prepare loaders to validate and test the models
    val_loader = ValDataLoader(data.validation, 256)
    test_loader = ValDataLoader(data.test, 256)

    # first of all, we run the grid searches for each model and each fold with seed equal to 0
    # set seed for reproduction of experiments
    np.random.seed(starting_seed)
    torch.manual_seed(starting_seed)
    # create folds with the seed
    p_dict = {p: data.get_user_movie_ratings(keep=p) for p in proportions}
    pool = Pool(os.cpu_count())
    if parallelize:
        # start grid searches in parallel
        pool.starmap(grid_search, [(model_name, p_dict[p], p, data, starting_seed)
                                   for model_name in models
                                   for p in p_dict])
    else:
        for model_name in models:
            for p in p_dict:
                grid_search(model_name, p_dict[p], p, data, starting_seed)
    
    # now that every grid search is finished, we can run all the other experiments
    for seed in range(starting_seed, starting_seed + n_runs):
        # set seed to reproducibility of results
        np.random.seed(seed)
        torch.manual_seed(seed)
        print("Start experiment with seed %d" % seed)
        # create the datasets with p % of training ratings
        print("Creating the datasets with the given proportions of training ratings - " + str(proportions))
        p_dict = {p: data.get_user_movie_ratings(keep=p) for p in proportions}

        if parallelize:
            pool.starmap(experiment_run,
                         [(seed, model, p_dict[p], p, data, val_loader, test_loader, test_metrics, results)
                          for model in models
                          for p in p_dict])
        else:
            for model in models:
                for p in p_dict:
                    experiment_run(seed, model, p_dict[p], p, data, val_loader, test_loader, test_metrics, results)

    # convert to normal dict to save data
    normal_results = {m: {seed: dict(results[m][seed]) for seed in dict(results[m])} for m in results}

    # save the dictionary as json in the results/ folder
    with open(report_save_path, 'w') as fp:
        json.dump(normal_results, fp, indent=4)


# function for single run
def experiment_run(seed, model, fold, fold_percentage, data, val_loader, test_loader, test_metrics, results):
    print("Start experiment for %s model with seed %d" % (model, seed))
    # set seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    with open("./saved_models/%s-%.1f.json" % (model, fold_percentage)) as f:
        conf = json.load(f)

    # configure model for training with randomly initialized weights and not previously loaded weights
    mf_model = MatrixFactorization(data.n_users, data.n_movies, conf["emb_size"], biased=conf["mf_bias"])
    optimizer = torch.optim.Adam(mf_model.parameters(), lr=conf["lr"], weight_decay=conf["reg_coeff"])
    trainer = MFTrainer(mf_model, optimizer) if model == "mf" else (
        LTNTrainerMF(mf_model, optimizer, conf["alpha"]) if model == "ltn" else
         LTNTrainerMFGenres(mf_model, optimizer, conf["alpha"], conf["p"]))

    # create loader for the given proportion
    train_loader = TrainingDataLoader(fold, conf["batch_size"]) if model == "mf" else (
        TrainingDataLoaderLTN(fold, conf["batch_size"]) if model == "ltn" else
        (TrainingDataLoaderLTNGenres(fold,
                                     data.get_user_genre_ratings_dict(),
                                     data.movie_to_genres, conf["batch_size"])))

    if not os.path.exists("./saved_models/%s-%.1f_seed-%d.pth" % (model, fold_percentage, seed)):
        # training of the model on the given proportion
        print("No model checkpoint found for %s with seed %d on dataset with %s percent of training "
              "ratings, starting training" % (model, seed, fold_percentage))

        print("Training %s on the dataset with %s percent of training ratings" % (model, fold_percentage))
        trainer.train(train_loader, val_loader, "hit@10", 200, 20, 1,
                      "./saved_models/%s-%.1f_seed-%d.pth" % (model, fold_percentage, seed))

    # load of the best model
    print("Loading best weights for %s trained on the dataset with %s percent of training ratings" % (
        model, fold_percentage))
    trainer.load_model("./saved_models/%s-%.1f_seed-%d.pth" % (model, fold_percentage, seed))

    # test of the model trained on the given proportion
    print("Test of %s trained on the dataset with %s percent of training ratings" % (model, fold_percentage))
    results[model][seed][fold_percentage] = trainer.test(test_loader, test_metrics)
            