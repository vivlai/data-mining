import scipy
import sys
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statistics

topicNum = 20
docNum = 7000
stdTheta = []
stdThetaList = []

basename = sys.argv[1]

#######
wordList = []
#######
count = {}
total = {}
countB = {}
flag = 0

doc_topic_stability = {}
for i in range(docNum):
    doc_topic_stability[i] = []

doc_topic_final = {}
for i in range(docNum):
    doc_topic_final[i] = np.zeros(topicNum)

for i in np.arange(1000, 2010, 10):
    filename = '%s%d.assign' % (basename, i)
    infile = open(filename, "r")
    docCount = 0
    for line in infile:

        doc_topic_dis = [0 for i in range(topicNum)]
        wordCount = 0

        for token in line.split()[2:]:

            parts = token.split(":")
            x = int(parts.pop())
            z = int(parts.pop())
            word = ":".join(parts)
            doc_topic_dis[z] += 1

            wordCount += 1

        doc_topic_final[docCount] += np.array(doc_topic_dis)

        for d in range(len(doc_topic_dis)):
            doc_topic_dis[d] = doc_topic_dis[d]/wordCount

        doc_topic_stability[docCount].append(doc_topic_dis)
        docCount += 1

    print(i)

for k,v in doc_topic_final.items():
    doc_topic_final[k] = doc_topic_final[k]/sum(doc_topic_final[k])
    '''
    if doc_topic_final[k][1] >= 0.1:
        print("************"+'doc'+str(k+1))
        for i in range(len(doc_topic_final[k])):
            if doc_topic_final[k][i] >= 0.1:
                print('topic'+str(i))
        print('**************')
    '''
f = open('./final_vector.txt','w+')
for i in range(docNum):
    tmpp = doc_topic_final[i]
    for j in tmpp:
        f.write(str(j)+' ')
    f.write('\n')
f.close()

for k,v in doc_topic_stability.items():
    doc_topic_stability[k] = np.array(doc_topic_stability[k])
    doc_topic_stability[k] = np.mat(doc_topic_stability[k])

for k,v in doc_topic_stability.items():
    stdTheta.append(v.std(0))
'''
for i in range(topicNum):
    print("topic"+str(i)+":", doc_topic_final[232][i])

print(stdTheta[33])
'''
'''
for t in range(docNum):

    #topic1 = doc_topic_stability[232][:,1]
    #topic10 = doc_topic_stability[232][:,7]
    #topic19 = doc_topic_stability[232][:,10]
    #topic20 = doc_topic_stability[232][:,19]
    
    topics = [];
    norm = []
    cc = 0
    
    for i in range(topicNum):
        thistopic = doc_topic_stability[t][:,i]
        tmp = []
        for j in thistopic:
            tmp.append(np.float64(j[0]))
        tmp = sorted(tmp)
        tmpNormlized = []
        if sum(tmp) == 0:
            norm.append(1)
        else:
            for j in tmp:
                #tmpNormlized.append(j/sum(tmp))
                tmpNormlized.append(j)
            tmpNormlized = np.array(tmpNormlized)
            norm.append(tmpNormlized.std())
    stdThetaList.append(norm)
'''
'''
storeVis = []
topic29 = doc_topic_stability[9318][:,24]
topic31 = doc_topic_stability[9318][:,39]
topic47 = doc_topic_stability[9318][:,48]
storeVis.append(topic29)
storeVis.append(topic31)
storeVis.append(topic47)

for i in range(3):
    tmp = []
    for j in storeVis[i]:
        tmp.append(np.float64(j[0]))
    tmp = sorted(tmp)
    storeVis[i] = tmp



for t in range(docNum):
    
    
    topics = [];
    norm = []
    
    for i in range(topicNum):
        thistopic = doc_topic_stability[t][:,i]
        tmp = []
        for j in thistopic:
            tmp.append(np.float64(j[0]))
        tmp = sorted(tmp)
        tmpNormlized = []
        if tmp[99] == 0:
            norm.append(1)
        else:
            for j in tmp:
                tmpNormlized.append(j/sum(tmp))
            tmpNormlized = np.array(tmpNormlized)
            norm.append(1+tmpNormlized[99])
    stdThetaList.append(norm)



f = open('./tuned.txt','w+')
for i in range(len(stdThetaList)):
    tmpp = stdThetaList[i]
    for j in tmpp:
        f.write(str(j)+' ')
    f.write('\n')
f.close()



cc = 0
for i in storeVis:
        tmp = i
        tmpNormlized = [float(j)/sum(tmp) for j in tmp]
        print(tmpNormlized)
        
        lowess = sm.nonparametric.lowess(tmpNormlized, range(101), frac=0.3)
        if cc == 0:
            plt.plot(lowess[:, 0], lowess[:, 1], label = str(25))
        if cc == 1:
            plt.plot(lowess[:, 0], lowess[:, 1], label = str(40))
        if cc == 2:
            plt.plot(lowess[:, 0], lowess[:, 1], label = str(49))
        
        cc += 1
        
plt.legend()
plt.show()
'''
