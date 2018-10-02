
import tweepy
import json
import time


consumer_key = 'YJictlVzU6fdw8ACHNADhDefK'
consumer_secret = 'N8SdsTCFkF44dJZf34Jet7b1HL92PJKkvsWwaJ8lJm0qG0uTHB'
access_token_key = '2454102925-uiOHmzODXxurqqM4QhRzOAZEW3JyYkAnD37ihED'
access_token_secret = 'WvmV0dafDbAG8lmIyFWJiSri1FB1YHwG7sQheXTLIxYLw'

# consumer_key='FJkeW0VV0D6HGPYlF5UfklTK5'
# consumer_secret='IM8zRQFIq4wbKBgikZKNLqiEkHH6ePSg20Ag6bE1QLY6dIQPGM'
# access_token_key='4921031892-twRpm76J6kgd3cWp2d4dIMkp674ocaggbQiUgCX'
# access_token_secret='nTPkL7TXTD4winCFu8INTzdE6ALAIYNk9Tb39d4R0DgYS'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token_key, access_token_secret)
myApi = tweepy.API(auth)

#tweet Collection

def print_info(tweet):
    print '***************************'
    print 'Tweet ID: ', tweet['id']
    print 'User Name: ', tweet['user']['screen_name']
    try:
	    print 'Tweet Text: ', tweet['text']
    except:
		pass

def CollectRestData():
    query = "asthma"
    GEO = "40.7127750,-74.0059730,30mi"  # NYC
    outputFile = open("restDataCollection.csv", "a+")

    # Collect most recent 100 tweets
    tweets = myApi.search(q=query, geocode=GEO, count=100)
    for tweet in tweets:
        outputFile.write(json.dumps(tweet._json) + "\n")

    MAX_ID = tweets[-1].id

    # Continue collecting tweets till last tweet
    while len(tweets) > 0:
        try:
            tweets = myApi.search(q=query, geocode=GEO, count=100, max_id=MAX_ID)
            if tweets:
                MAX_ID = tweets[-1].id
                print MAX_ID, len(tweets)
                for tweet in tweets:
                    outputFile.write(json.dumps(tweet._json) + "\n")

        except tweepy.TweepError:
            print('exception raised, waiting for 15 minutes')
            time.sleep(10 * 60)
            break


# def writeCSV():
#     query = "asthma"
#     GEO = "40.7127750,-74.0059730,30mi"  # NYC
#     tweets = myApi.search(q=query, geocode=GEO, count=100)
#
#     for tweet in tweets:
#         print tweet.user.screen_name
#         data = {"screen_name": tweet.user.screen_name}
#         with open("Filtered.txt", 'a+') as data_2:
#             data_2.write(json.dumps(str(data)))
#             data_2.write("\n")

def writeCSV():

    fileOpen = open("restDataCollection.csv","r")
    fileWrite = open("FilteredTweets.csv","w")
    fileWrite.write('TWEET_ID$$username$$text$$matched$$positive'+'\n')

    while 1:
        readlines = fileOpen.readline()
        print("lines",readlines)
        readlines = readlines.replace('\n','')
        readlines = readlines.replace('\n\n', '')
        if not readlines:
            break
        print("read:",readlines)
        readTweet = json.loads(readlines)
        print("tweet:",readTweet)
        readTweet = json.loads(json.dumps(readTweet))
        print("tweet2",readTweet)
        # print(readTweet)
        tweet_id = readTweet['id']
        username = readTweet['user']['screen_name']
        username=username.encode('ascii','ignore')
        print("tweetid:",tweet_id)
        print("username:",username)
        try:
            text = readTweet['text']
            text=text.encode('ascii','ignore')
        except:
            pass
        matched = "True"

        myLine = "$$".join([str(tweet_id),username,text,str(matched)])+'\n'
        print("myLine:",myLine)
        fileWrite.write(myLine)
    fileWrite.close()
    fileOpen.close()



if __name__ == '__main__':

    CollectRestData()
    writeCSV()
    pass


