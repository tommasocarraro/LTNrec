# LTNrec
Application of Logic Tensor Networks to Recommender Systems.

The idea is to use the rich information provided by the [MindReader](https://mindreader.tech/dataset/) dataset to construct logical formulas useful to learn a Logic Tensor Network.

The aim of this project is to prove that the availability of background knowledge is crucial when training data is scarce. 

For the MindReader dataset, the background knowledge can be built based on items' side information, for instance:
1. likes of the users for the genres of the movies;
2. likes of the users for the actors, directors;
3. likes of the users for the decades or movie companies.

The experiment aims at reducing the information about the movie ratings of the MindReader dataset (making the dataset more sparse) and showing that the addition of background knowledge is helpful when the sparsity increases.
