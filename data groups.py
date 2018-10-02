# Step 1: Identify 100 labeled users in each of the six age groups,
#         Groups: 18-24, 25-34, 35-44, 45-54, 55-64, 64+

import re
from datetime import datetime
import json

date = datetime.now()
current_year = date.year

group_13_18 = []
group_18_24 = []
group_25_34 = []
group_35_44 = []
group_45_54 = []
group_55_64 = []
group_64_plus = []

group_13_18_tweets = []
group_18_24_tweets = []
group_25_34_tweets = []
group_35_44_tweets = []
group_45_54_tweets = []
group_55_64_tweets = []
group_64_plus_tweets = []

def age_and_year(age_data, year_data):
    number = [int(s) for s in re.findall(r'\b\d+\b', age_data)]
    age = [n for n in number if len(str(n)) == 2]
    yr = [int(s) for s in re.findall(r'\b\d+\b', year_data)]

    return int(age[0]), int(yr[5])


def assign_age_group_and_content(current, user_name, twt_data, twt_description, Screen_name, Date):
    dict = {}
    dict['name'] = user_name
    dict['tweet'] = twt_data
    dict['created_at'] = Date
    dict['screen_name'] = Screen_name

    if current in range(13, 19):
        group_13_18_tweets.append(json.dumps(dict))
    elif current in range(18, 25):
        group_18_24_tweets.append(json.dumps(dict))
    elif current in range(25, 35):
        group_25_34_tweets.append(json.dumps(dict))
    elif current in range(35, 45):
        group_35_44_tweets.append(json.dumps(dict))
    elif current in range(45, 55):
        group_45_54_tweets.append(json.dumps(dict))
    elif current in range(55, 65):
        group_55_64_tweets.append(json.dumps(dict))
    elif current > 64:
        group_64_plus_tweets.append(json.dumps(dict))

def upload_data(filename, data):
    f = open(filename, "a+")
    for item in data:
        f.writelines(item.encode("utf-8") + '\n')

def print_groups():
    print "\nLength of Group_13_18:", len(group_13_18)
    print "Length of Group_18_24:", len(group_18_24)
    print "Length of Group_25_34:", len(group_25_34)
    print "Length of Group_35_44:", len(group_35_44)
    print "Length of Group_45_54:", len(group_45_54)
    print "Length of Group_55_64:", len(group_55_64)
    print "Length of Group_64_+ :", len(group_64_plus)


age_keywords = ["I'M", "I'm", "I am", "i'm", "i am", "my age is", "hours till I'm", "within days I'm", "ho anni",
                "me old", "months till am", "I cannot wait till", "I look like", "years already!", "birthday in", "in few months"]

input_file_data = open("data/2013-09.txt", "r")
for line in input_file_data:
    try:
        twt = (json.loads(line))
        name = twt['user']['name']
        screen_name = twt['user']['screen_name']
        dt = twt['created_at']
        flag = 0
        for key in age_keywords:
            if key in twt['text']:
                flag = 1
                break

        if flag == 1:
            twt_age, twt_year = age_and_year(twt['text'], twt['created_at'])
            current_age = (current_year - twt_year) + twt_age
            assign_age_group_and_content(current_age, name.encode("utf-8"), twt['text'], twt['user']['description'],
                                         screen_name, dt)

            # print "age @ tweet time:", twt_age, "\n year of tweet:", twt_year
            # print "current_age", current_age, name, "\n\n"
    except:
        continue

print_groups()
upload_data("group_13_18_tweet_n_date_info.txt", group_13_18_tweets)
upload_data("group_18_24_tweet_n_date_info.txt", group_18_24_tweets)
upload_data("group_25_34_tweet_n_date_info.txt", group_25_34_tweets)
upload_data("group_35_44_tweet_n_date_info.txt", group_35_44_tweets)
upload_data("group_45_54_tweet_n_date_info.txt", group_45_54_tweets)
upload_data("group_55_64_tweet_n_date_info.txt", group_55_64_tweets)
upload_data("group_64_plus_tweet_n_date_info.txt", group_64_plus_tweets)


"""
07  Length of Group_18_24: 26377
    Length of Group_25_34: 16472
    Length of Group_35_44: 3433
    Length of Group_45_54: 1564
    Length of Group_55_64: 1573
    Length of Group_64_+ : 2565
    
08  Length of Group_18_24: 25633
    Length of Group_25_34: 13207
    Length of Group_35_44: 3012
    Length of Group_45_54: 1651
    Length of Group_55_64: 1527
    Length of Group_64_+ : 2887
    
09  Length of Group_18_24: 24581
    Length of Group_25_34: 11696
    Length of Group_35_44: 2571
    Length of Group_45_54: 1467
    Length of Group_55_64: 1377
    Length of Group_64_+ : 2293	
"""