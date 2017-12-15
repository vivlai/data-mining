import json
import csv
import argparse
from sets import Set
from collections import defaultdict

# This program select subset of users and all related reviews

def find(review, userids):
    return review['user_id'] in userids
def getStars(cnt,summ,x,y):
        if cnt[(x,y)] > 0:
            return summ[(x,y)]/cnt[(x,y)]
        return -1
 
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Preprocessing of project')

    parser.add_argument('--business', type=str,
                help="business json",
                default="business.json",
                required=False)
    parser.add_argument('--users', type=str,
                help="output user data file name",
                default="user_selected.json",
                required=False)
    parser.add_argument("--reviews", type=str,
                help="output reviews file name",
                default="reviews_selected.json",
                required=False)
    parser.add_argument("--user_review_collection", type=str,default="user_star.data",required=False)
    args = parser.parse_args()

    restaurants = {}
    with open(args.business) as f:
        for line in f:
            data = json.loads(line)
            restaurants[data['business_id']] = data['categories']
    print("All business done")

    users = []
    userids = Set()
    totalReviews = 0
    with open(args.users) as json_file:
        for line in json_file:
            data = json.loads(line)
            users.append(data)
            userids.add(data['user_id'])

    reviews = []
    stars_cnt = defaultdict(lambda:0)
    stars_sum = defaultdict(lambda:0)
    category_sum = defaultdict(lambda:0)
    busids = Set()
    categories = Set()
    with open(args.reviews) as json_file:
        for line in json_file:
            data = json.loads(line)
            reviews.append(data)
            busids.add(data['business_id'])
            stars_cnt[(data['user_id'],data['business_id'])] += 1
            stars_sum[(data['user_id'],data['business_id'])] += data['stars'] 
            for c in restaurants[data['business_id']]:
                categories.add(c)
                category_sum[(data['user_id'],c)] += data['stars']
    
    print('user count {}'.format(len(userids)))
    print('restaurant count {}'.format(len(busids)))
    print('categories count {}'.format(len(categories)))
    nowuser = 0

    user_cate = defaultdict(lambda:0)
    user_rest = defaultdict(lambda:0)
    with open(args.user_review_collection, 'w') as output:
        output.write("user-rest\n")
        for u, b in stars_sum:
            output.write(u)
            output.write(',')
            output.write(b)
            output.write(',')
            output.write(str(stars_sum[(u,b)]))
            output.write(',')
            output.write(str(stars_cnt[(u,b)]))
            output.write("\n")
            user_rest[u] += 1
        output.write("user-category\n")
        for u, c in category_sum:
            output.write(u)
            output.write(',')
            output.write(c)
            output.write(',')
            output.write(str(category_sum[(u,c)]))
            output.write('\n')
            user_cate[u] += 1

    ss = 0
    for u in user_cate:
        ss += user_cate[u]
    ss /= len(user_cate)
    print('avg users have {} categories'.format(ss))
    ss = 0
    for u in user_rest:
        ss += user_rest[u]
    ss /= len(user_rest)
    print('avg users have {} restaurants'.format(ss))
    '''
        fieldnames = ['user_id']
        for busid in busids:
            fieldnames.append(busid)
        print("Begin processing datas:")
        writer = csv.DictWriter(output, fieldnames)
        print("Begin writing headers:")
        writer.writeheader()
        print("Headers done")
        for userid in userids:
            dd = {'user_id':userid}
            nowuser += 1
            print(nowuser)
            for busid in busids:
               dd[busid] = getStars(stars_cnt,stars_sum,userid,busid)
            writer.writerow(dd)
            '''
