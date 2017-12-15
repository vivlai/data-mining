import json
import string
from nltk.corpus import stopwords
from nltk.tag import pos_tag

'''
1. change the direction here
2. the output files are: (1) pre-processed reviews (2) the corresponded star ratings
3. puncuations, NNPs, digits and stop words are eliminated
'''

viewData = []

with open("/Users/xinglinzi/Downloads/dataset/Yelp500.json") as json_file:
    for line in json_file:
        json_data = json.loads(line)
        viewData.append(json_data)

stars = []; reviews = []
for i in range(len(viewData)):
    stars.append(viewData[i]['stars'])
    reviews.append(viewData[i]['text'])

exclude = set(string.punctuation)
for i in range(len(reviews)):
    s = reviews[i]
    s = ''.join(ch for ch in s if ch not in exclude)
    sList = s.strip().split()
    tagged_sent = pos_tag(sList)
    filtered0 = [word for word,pos in tagged_sent if pos != 'NNP']
    filtered0 = [word.lower() for word in filtered0]
    filtered1 = [word for word in filtered0 if word not in stopwords.words('english')]
    #filtered2 = [word for word in filtered1 if len(word) >= 2]
    filtered3 = [word for word in filtered1 if not word.isdigit()]
    reviews[i] = filtered3

f = open("./docrates.txt",'w+')
for i in stars:
    f.write(str(i) + '\n')
f.close()

f = open("./input.txt",'w+')
for i in reviews:
    f.write('')
    for word in i:
        f.write(word+' ')
    f.write("\n")
f.close()

