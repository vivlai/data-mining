import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn import linear_model
import scipy.stats as stats
import math


allindex = []

c = 0
for line in open('./docrates.txt'):
    allindex.append(line.strip().split()[0])

final_vec = []
for line in open('/Users/xinglinzi/Downloads/topic_vis/lda/final_vector.txt'):
    l = line.strip().split(' ')
    for i in range(len(l)):
        l[i] = float(l[i])
    final_vec.append(l)


#X_train, X_test, y_train, y_test = train_test_split(final, allindex, test_size=0.2, random_state=0)
#logreg = LogisticRegression(multi_class='ovr')
#logreg = MultinomialNB(alpha=.01)

logreg = svm.LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
     verbose=0)

#logreg.fit(X_train, y_train)

scores1 = cross_val_score(logreg, final_vec, allindex, cv=10)

print(np.mean(scores1))

for i in range(5000):
    final_vec[i] = np.array(final_vec[i]).astype(np.float)
    allindex[i] = float(allindex[i])
regr = linear_model.LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(final_vec, allindex, test_size=0.2, random_state=0)
regr.fit(X_train, y_train)
prediction = regr.predict(X_test)
rho1, p1 = stats.pearsonr(y_test,prediction)
rho2, p2 = stats.spearmanr(y_test,prediction)
rho3, p3 = stats.kendalltau(y_test,prediction)
print(rho1,rho2,rho3)
