import json
import argparse
from sets import Set

# This program select subset of users and all related reviews

def find(review, userids):
    return review['user_id'] in userids

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Preprocessing of project')
    parser.add_argument('--num', type=int,
                    help='number of users, 500 by default',
                    default=1000)

    parser.add_argument('--users', type=str,
                help="output user data file name",
                default="user_selected.json",
                required=False)
    parser.add_argument("--reviews", type=str,
                help="output reviews file name",
                default="reviews_selected.json",
                required=False)
    args = parser.parse_args()

    users = []
    userids = Set()
    totalReviews = 0
    with open("user.json") as json_file:
        cnt = 0
        for line in json_file:
            if cnt >= args.num:
                break
            data = json.loads(line)
            users.append(data)
            #print(data['user_id'])
            userids.add(data['user_id'])
            cnt += 1
            totalReviews += data['review_count']
    print(totalReviews)
    print(len(users))
 
    with open(args.users,'w') as f:
        for user in users:
            f.write(json.dumps(user))
            f.write("\n")

    reviewCnt = 0
    reviews = []
    with open("review.json") as json_file:
        for line in json_file:
            data = json.loads(line)
            if find(data, userids):
                reviews.append(data)
                reviewCnt += 1
                #print ('ok')
    print(reviewCnt)
    with open(args.reviews,'w') as f:
        for review in reviews:
            f.write(json.dumps(review))
            f.write("\n")
   
