# task 1: Train a SVM classifier based on the training set enriched with historical tweets and
# report 10 fold cross validation accuracy

import numpy as np
import pandas as pd
from TwitterAPI import TwitterAPI
from datetime import datetime

from sklearn import svm
from sklearn.feature_extraction.text import *
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

t1 = datetime.now()
api = TwitterAPI('', '', '', '')

stopwords = []
sw = open("stopwords_en.txt", "r")
for word in sw:
    stopwords.append(word.rstrip())

male_keywords = []
m_keys = open("Male_Frequent_words.txt", "r")
for word in m_keys:
    male_keywords.append(word.rstrip())

female_keywords = []
f_keys = open("Female_Frequent_words.txt", "r")
for word in f_keys:
    female_keywords.append(word.rstrip())


def stopwords_removal(current_twt):
    updated_twt = ' '
    current_twt = current_twt.lower()
    for twt_word in current_twt.encode("utf-8").split():
        if twt_word not in stopwords:
            updated_twt = updated_twt + twt_word + ' '
    return updated_twt


def identification(current_user):
    male_cnt = 0
    female_cnt = 0
    for item in current_user.split():
        if item in male_keywords:
            male_cnt += 1
        if item in female_keywords:
            female_cnt += 1

    return male_cnt, female_cnt

user_names = []
f = open("user_names.txt", "r")
for line in f:
    user_names.append(line.rstrip())

count = 0
tweets = []
cls_label = []
for name in user_names:
    if count < 450:
        count += 1
        try:
            results = api.request('statuses/user_timeline', {'screen_name': name, 'count': 200})

            # Creation of X
            User_tweets = ' '
            for result in results:
                User_tweets = User_tweets + stopwords_removal(result['text']) + ', '

            if len(User_tweets) != 0:
                tweets.append(User_tweets)

                # Creation of y
                male_count, female_count = identification(User_tweets)
                if male_count > female_count:
                    cls_label.append(1)  # print "male"
                else:
                    cls_label.append(0)  # print "female"
        except:
            print name
    else:
        break

X = np.array(tweets)
y = np.array(cls_label)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X).toarray()
X = pd.DataFrame(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print "X_train.shape", X_train.shape, "  y_train.shape", y_train.shape
print "X_test.shape", X_test.shape, "  y_test.shape", y_test.shape

svc = svm.SVC(kernel='linear', C=10)
SVM_y = svc.fit(X_train, y_train)

predicted_label = svc.predict(X_test)
print "\n\n Accuracy_Score: ", accuracy_score(y_test, predicted_label)

t2 = datetime.now()
print "\n Run time: ", t2 - t1
