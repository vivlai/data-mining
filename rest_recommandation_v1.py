import json
import csv
import math
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

def find_similar_people(me, user_cate):
    similarity = defaultdict(lambda:0.0) # user -> cosine product
    lenth_u = defaultdict(lambda:0) # user -> sigma(c_i^2)
    for u,c in user_cate:
        if me != u:
            if (me,c) in user_cate:
                    similarity[u] += user_cate[(u,c)] * user_cate[(me,c)]
        lenth_u[u] += user_cate[(u,c)]*user_cate[(u,c)]
    for u in similarity:
        similarity[u] /= math.sqrt(lenth_u[u])*math.sqrt(lenth_u[me])

    array = []
    for u in similarity:
        array.append((-similarity[u],u))
    array.sort()
    return array
 
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
    parser.add_argument("--base", type=int,default=200,required=False)
    args = parser.parse_args()

    restaurants = []
    with open(args.business) as f:
        for line in f:
            data = json.loads(line)
        restaurants.append( data['categories'])
        print("All business done")


    # Let user_rest be a mapping : { (user_id,business_id) -> avg_star  }
    # Let user_cate be a mapping : { (user_id,category) -> total_star }
    # total_star because we think user is perfer a certain category if he/she has high total star in this category

    # Given a user vector:(avg_star_of_rest_1, avg_star_of_rest_2, ... , avg_star_of_rest_n) and (total_star_of_category_1, total_star_of_category_2, ..., total_star_of_category_m)
    # we find out users that similar to he/her by:
    # find list of [user1,user2,user3...userK] that maximize cosine product using user_category vector
    # fetch out user_restaurant vector from [user1,...,userK]
    # calulate sum star of each restaurants by sum up avg_star together
    # select the best one

    user_rest = defaultdict(lambda:0) # (user,rest) -> star
    user_rest_vec = defaultdict(lambda:[]) # user -> (restaurant,star)
    user_cate = defaultdict(lambda:0) # (user,category) -> star
    user_list = Set() 
    restaurant_list = Set()
    category_list = Set()

    with open(args.user_review_collection,'r') as f:
        for line in f:
            a = line.split(',')
            if len(a) == 4: # user, rest, ave_star, cnt_
                user_rest[(a[0],a[1])] = int(a[2])
                user_rest_vec[a[0]].append((a[1],a[2]))
                user_list.add(a[0])
                restaurant_list.add(a[1])
            elif len(a) == 3: # user, category, total_star
                user_cate[(a[0],a[1])] = int(a[2])
                category_list.add(a[1])
    f = True
    stats = [[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]
    stats2 = [0.0] * 12 
    cc = 0


    rmse = 0.0
    testN = 200
    rmseN = 0
    hitRate = 0.0

    stats_hit = [0,0,0,0,0,0]

    for u in user_list:
        cc += 1
        print (cc)
        if cc > testN:
            break
        # Find similar people based on the user-category vector
        # peopleList is an array of (-similarity, user) sorted by similarity for descending order
        peopleList = find_similar_people(u, user_cate)
        expectation_star = defaultdict(lambda:[0.0,0.0]) # {restaurant -> star}
        for weight, user in peopleList[:args.base]:
            for restaurant, star in user_rest_vec[user]:
                expectation_star[restaurant][0] -= float(star)*weight
                expectation_star[restaurant][1] -= weight

        ranking = []
        for rest in expectation_star:
            expectation_star[rest][0] /= expectation_star[rest][1]
            if (u,rest) in user_rest:
                diff = expectation_star[rest][0] - user_rest[(u,rest)]
                rmse += diff*diff
                rmseN += 1
                stats2[int(round(diff))] += 1
                #userLocation = (restaurants[rest]['latitude'], restaurants[rest]['longitude'])

            ranking.append((expectation_star[rest][0],rest))
        ranking.sort()
        ranking.reverse()

        #cnt = 3
        #print(ranking)
        #hit = 0
        #hitN = 0
        for rate, rest in ranking[:50]:
            if (u, rest) in user_rest :
                stats_hit[user_rest[(u,rest)]] += 1
            #cnt -= 1
            #stats[cnt][user_rest[u,rest]] += 1
        #hitRate += float(hit)/hitN
    #print(stats[2])
    #print(stats[1])
    #print(stats[0])
    rmse /= rmseN
    rmse = math.sqrt(rmse)
    print(rmse)
    print(stats_hit)
    s = sum(stats2)
    for i in range(len(stats2)):
        stats2[i] /= s
    print(stats2) # 10-len array

    

