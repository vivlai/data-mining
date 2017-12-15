from gensim.models import doc2vec
from collections import namedtuple
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
import numpy as np
import scipy.stats as stats
from sklearn import linear_model
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer


def content():
    data_list = []

    for line in open('./input.txt'):
        data_list.append(line)

    LabelDoc = namedtuple('LabelDoc','words tags')
    #exclude = set(string.punctuation)

    all_docs = []
    count = 0

    for sen in data_list:
        tag = ['SEN_'+str(count)]
        count += 1
        #sen = ''.join(ch for ch in sen if ch not in exclude)
        #senn = ''.join(word+' ' for word in sen.split(' ') if word not in stopList)
        all_docs.append(LabelDoc(sen.split(),tag))


    return all_docs, data_list

if __name__ == '__main__':

    ##########load LDA data###############

    final_vec = []
    for line in open('/Users/xinglinzi/Downloads/topic_vis/lda/final_vector.txt'):
        l = line.strip().split(' ')
        for i in range(len(l)):
            l[i] = float(l[i])
        final_vec.append(l)

    ##########call para2vec###############
    all_docs, doc_list = content()
    model = doc2vec.Doc2Vec(size=100, window=3, alpha=0.025,min_alpha=0.025, min_count=3)
    model.build_vocab(all_docs)

    for epoch in range(10):
        model.train(all_docs)
        model.alpha -= 0.002
        model.min_alpha = model.alpha

    model.save('model.doc2vec')

    ##########build Tf-idf###############

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(doc_list)
    tfidf = X.toarray()
    #tfidf = list(tfidf)
    pca = PCA(n_components=100)
    pca.fit(tfidf)
    tfidf_pca = pca.transform(tfidf)

    doc2vec_all = []
    for i in range(7000):
        doc2vec_all.append(list(model.docvecs[i]))

    ##########################
    mingle = []
    for i in range(7000):
        mingle.append(final_vec[i]+list(tfidf_pca[i]))

    ##########################

    allindex = []
    for line in open('./docrates.txt'):
        allindex.append(line.strip().split()[0])

    logreg1 = LogisticRegression(multi_class='ovr')
    #logreg = MultinomialNB(alpha=.01)

    logreg2 = svm.LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
        intercept_scaling=1, loss='squared_hinge', max_iter=1000,
        multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
        verbose=0)

    scores1 = cross_val_score(logreg1, mingle, allindex, cv=10)
    scores2 = cross_val_score(logreg2, mingle, allindex, cv=10)
    #classification accuracy
    print(np.mean(scores1),np.mean(scores2))

    for i in range(7000):
        #doc2vec_all[i] = np.array(doc2vec_all[i]).astype(np.float)
        mingle[i] = np.array(mingle[i]).astype(np.float)
        allindex[i] = float(allindex[i])
    regr = linear_model.LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(mingle, allindex, test_size=0.2, random_state=0)
    regr.fit(X_train, y_train)
    prediction = regr.predict(X_test)
    rho1, p1 = stats.pearsonr(y_test,prediction)
    rho2, p2 = stats.spearmanr(y_test,prediction)
    rho3, p3 = stats.kendalltau(y_test,prediction)
    print(rho1,rho2,rho3)
    #prediction correlation

    pca = PCA(n_components=2)
    pca.fit(mingle)
    transform_data = pca.transform(mingle)

    starDict = {}
    for i in range(1,6):
        starDict[i] = [[],[]]

    for i in range(7000):
        rate = allindex[i]
        point = transform_data[i]
        starDict[rate][0].append(point[0])
        starDict[rate][1].append(point[1])


    plt.scatter(starDict[1][0],starDict[1][1],color='r',s=2,label = '1')
    #plt.scatter(starDict[2][0],starDict[2][1],color='b',s=2,label = '2')
    #plt.scatter(starDict[3][0],starDict[3][1],color='y',s=2,label = '3')
    #plt.scatter(starDict[4][0],starDict[4][1],color='k',s=2,label = '4')
    plt.scatter(starDict[5][0],starDict[5][1],color='c',s=2,label = '5')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(loc='1')
    plt.show()




