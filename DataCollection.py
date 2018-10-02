import tweepy
import json
import time

consumer_key = 'YJictlVzU6fdw8ACHNADhDefK'
consumer_secret = 'N8SdsTCFkF44dJZf34Jet7b1HL92PJKkvsWwaJ8lJm0qG0uTHB'
access_token_key = '2454102925-uiOHmzODXxurqqM4QhRzOAZEW3JyYkAnD37ihED'
access_token_secret = 'WvmV0dafDbAG8lmIyFWJiSri1FB1YHwG7sQheXTLIxYLw'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token_key, access_token_secret)
myApi = tweepy.API(auth)

class StreamListener(tweepy.StreamListener):
    def on_data(self, raw_data):
        try:
            jdata = json.loads(str(raw_data))
            outputFile = open("streamData1.txt", "a+")
            outputFile.write(json.dumps(jdata) + "\n")
            outputFile.close()
        except:
            print 'Data writting exception.'

def CollectStreamData():
    while(True): 
        sl = StreamListener()
        stream = tweepy.Stream(auth, sl)
        try: 
            stream.filter(track = ['asthma'])
        except:
            print 'Exception occur!'
            
def CollectRestData():
    query = "asthma" 
    GEO = "40.7127750,-74.0059730,30mi" #NYC 
    outputFile = open("restData1.txt", "a+")
    
    #Collect most recent 100 tweets
    tweets = myApi.search(q=query, geocode=GEO, count=100)
    for tweet in tweets:
        outputFile.write(json.dumps(tweet._json) + "\n")
    
    MAX_ID = tweets[-1].id
    
    #Continue collecting tweets till last tweet    
    while len(tweets) > 0:
        try:
            tweets = myApi.search(q=query, geocode=GEO, count=100, max_id = MAX_ID)
            if tweets:
                MAX_ID = tweets[-1].id
                print MAX_ID, len(tweets)
                for tweet in tweets:
                    outputFile.write(json.dumps(tweet._json) + "\n")
    
        except tweepy.TweepError:
            print('exception raised, waiting for 15 minutes')
            time.sleep(10*60)
            break
              
if __name__ == '__main__':
    #Collect tweets using Stream API
    #CollectStreamData()
    
    #Collect tweets using REST API
    CollectRestData()



