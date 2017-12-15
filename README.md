# data-mining project

1. readJson.py: This script is used for us to read certain size of reviews json data and do pre-processing(puncuations, NNPs, digits and stop words are eliminated).

2. topwords_posterior_doc.py: This script is used for us to deal with the outputs of the Topic Model java script which is adopted from Dr. Michael J. Paul(http://cmci.colorado.edu/~mpaul/downloads/mftm.php). After run of this script, we can get the LDA embedding of each review.

3. combination.py: Covering TF-IDF, Paragragh2Vec (Gensim.models.doc2vec) and LDA. This script implements the combination of possible vectors and run the experiements(classification, prediction, clustering).

4. LDA.py: Getting the LDA vector.
