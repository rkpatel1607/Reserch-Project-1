import pandas as pd
from TwitterAPI import TwitterAPI
from django.utils.encoding import smart_str

api = TwitterAPI('#', '#', '#', '#')

Data = pd.read_csv("gender-classifier-DFE-791531.csv")

for name in Data['name']:
    fileName = name + ".txt"
    outputFile = open(fileName, "a+")
    tweets = []

    results = api.request('statuses/user_timeline', {'screen_name': name, 'count': 200})

    for i in results:
        twt = smart_str(i['text'])
        # print twt
        tweets.append(twt)
        twt_id = i['id'] - 1

    while len(tweets) > 0:
        results = api.request('statuses/user_timeline', {'screen_name': name, 'count': 200, 'max_id': twt_id})

        for i in results:
            tweets.append(i['text'])
            # print i['text']
            twt_id = i['id']
        twt_id -= 1
        # print(len(tweets))
    break
