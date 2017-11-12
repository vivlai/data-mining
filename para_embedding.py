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


    return all_docs

if __name__ == '__main__':

    all_docs = content()
    model = doc2vec.Doc2Vec(size=100, window=3, alpha=0.025,min_alpha=0.025, min_count=3)
    model.build_vocab(all_docs)

    for epoch in range(10):
        model.train(all_docs)
        model.alpha -= 0.002
        model.min_alpha = model.alpha

    model.save('model.doc2vec')

    doc2vec_all = []
    for i in range(5000):
        doc2vec_all.append(model.docvecs[i])

    allindex = []
    for line in open('./docrates.txt'):
        allindex.append(line.strip().split()[0])

    logreg = LogisticRegression(multi_class='ovr')
    #logreg = MultinomialNB(alpha=.01)
    '''
    logreg = svm.LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
        intercept_scaling=1, loss='squared_hinge', max_iter=1000,
        multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
        verbose=0)
    '''
    scores = cross_val_score(logreg, doc2vec_all, allindex, cv=10)
    #classification accuracy
    print(np.mean(scores))

    for i in range(5000):
        doc2vec_all[i] = np.array(doc2vec_all[i]).astype(np.float)
        allindex[i] = float(allindex[i])
    regr = linear_model.LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(doc2vec_all, allindex, test_size=0.2, random_state=0)
    regr.fit(X_train, y_train)
    prediction = regr.predict(X_test)
    rho1, p1 = stats.pearsonr(y_test,prediction)
    rho2, p2 = stats.spearmanr(y_test,prediction)
    rho3, p3 = stats.kendalltau(y_test,prediction)
    print(rho1,rho2,rho3)
    #prediction correlation
    
    #PCA, project to 2D scatter plots
    pca = PCA(n_components=2)
    pca.fit(doc2vec_all)
    transform_data = pca.transform(doc2vec_all)

    starDict = {}
    for i in range(1,6):
        starDict[i] = [[],[]]

    for i in range(5000):
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
