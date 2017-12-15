# data-mining project

1. readJson.py: This script is used for us to read certain size of reviews json data and do pre-processing(puncuations, NNPs, digits and stop words are eliminated).

2. topwords_posterior_doc.py: This script is used for us to deal with the outputs of the Topic Model java script which is adopted from Dr. Michael J. Paul(http://cmci.colorado.edu/~mpaul/downloads/mftm.php). After run of this script, we can get the LDA embedding of each review.

3. combination.py: Covering TF-IDF, Paragragh2Vec (Gensim.models.doc2vec) and LDA. This script implements the combination of possible vectors and run the experiements(classification, prediction, clustering).

4. LDA.py: Getting the LDA vector.

5. scratch.ipynb: This notebook is used to run TF-IDF for clustering and classification tasks. It's a scratch book to run different sets of data on same tasks.

6. preprocess_select_users_reviews.py: select users and their reviews

7. preprocess_fetch_stars.py: make a user-restaurant table

8. rest_recommandation_v1.py: recommand based on 'similar' people's preferences 
