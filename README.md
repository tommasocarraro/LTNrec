# LTNrec

This is the code for the paper: Logic Tensor Networks for Top-N Recommendation.
It is an application of the Logic Tensor Networks Neuro-Symbolic framework to Recommender Systems. 

The idea is to use the rich information provided by the [MindReader](https://mindreader.tech/dataset/) dataset to construct logical formulas useful to learn a Logic Tensor Network.

The aim of the project is to show that the injection of background knowledge is crucial when training data is scarce. 

For the MindReader dataset, the background knowledge can be built based on items' content information, for instance:
1. preferences of the users for the genres of the movies (i.e., the information used in the paper);
2. preferences of the users for the actors, directors;
3. preferences of the users for the decades or movie companies.

The experiment aims at reducing the movie ratings of the MindReader dataset (making the dataset more sparse) and showing that the addition of background knowledge is helpful when the sparsity increases.

## How to run?

Simply run the `main.py` file in the root directory of the repository. The file runs a complete experiment:
1. it loads the dataset;
2. it creates 5 additional datasets with different sparsity, starting from the original one;
3. it trains 3 models (i.e., MF, LTN, and LTN with genres). For each model, it runs the experiment two times with different seeds. Only for the first execution, a grid search for parameter selection is performed;
4. it computes test metrics (i.e., hit@10, ndcg@10) and average the results across the runs;
5. it creates a LaTeX table containing the obtained results.
