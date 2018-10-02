# -*- coding: utf-8 -*-

import datetime
import io
import sys
from imp import reload

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import svm

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import sys
from sklearn.metrics import auc, roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

reload(sys)
#sys.setdefaultencoding('utf8')
import tweepy
import json
import numpy as np
import csv
from TwitterAPI import TwitterAPI
import time

api = TwitterAPI('6hckfkSBDEUJRxPATcCtcsdOp',
            		'GduOLco0wL1pOTboeClVPmZe8PPXEYD0VKnmKWpb2of7UNTbrY',
                    '399687253-xb9c6wdOQMBU8K2Jk3utxDLuqC21qRLTajkV9iel',
                    'bXdT9ejraFk6O9bNExA2sD71jgixMFby8nqzQwfsTLXKa')


stopwords = ["a", "about", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "around", "as", "at", "back", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both","bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on","once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thick", "thin", "third", "this", "those", "though", "three", "through", "throughout","thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves",
              '@', '#', '...', '....', '+', '-', 'rt', ',', '.',  ':)', 'w/', 'http', '✨']



# load usernames from organization and user lists files
def load_screenNames1(infile):
    screenNames = []
    for line in open(infile).readlines():
        items = line.strip().split(',')
        screenNames.append(str(items[1]))

    screenNames[0] = screenNames[0].replace(u'\ufeff', '')

    # print(len(screenNames), screenNames[:3])
    return list(set(screenNames))


def load_screenNames2(infile):
    screenNames = []
    for line in open(infile).readlines():
        items = line.strip().split(',')
        screenNames.append(str(items[0]))

    screenNames[0] = screenNames[0].replace(u'\ufeff', '')

    # print(len(screenNames), screenNames[:3])
    return list(set(screenNames))



# get each user historic tweets
def get_history_tweets(screenNames, outfile):

    print("length of screenNames: ", len(screenNames))
    tweets = []
    count2 = 0
    for screenName in screenNames:
        # print("userID: ", userID)
        tweet = []
        try:
            results = api.request('statuses/user_timeline',
                                  {"screen_name": screenName, 'count': 5, 'include_rts': False})
            count2 += 1
            for item in results:
                with open(outfile, 'a') as f:
                    f.write(json.dumps(item))
                    f.write('\n')

        except:
            continue

        if count2 >= 100:
            break



# # labels: individual: 0  organization: 1
# def save_library_results(infile1, infile2, outfile):
#     indId_predictY = {}
#     with open(infile1) as tsvfile:
#         reader = csv.reader(tsvfile, delimiter='\t')
#         for row in reader:
#             if row[1] == 'per':
#                 indId_predictY[row[0]] = int(0)
#             elif row[1] == 'org':
#                 indId_predictY[row[0]] = int(1)
#     # print("length of indId_predictY: ", len(indId_predictY))
#
#     orgId_predictY = {}
#     with open(infile2) as tsvfile:
#         reader = csv.reader(tsvfile, delimiter='\t')
#         for row in reader:
#             if row[1] == 'per':
#                 orgId_predictY[row[0]] = int(0)
#             elif row[1] == 'org':
#                 orgId_predictY[row[0]] = int(1)
#     # print("length of orgId_predictY: ", len(orgId_predictY))
#
#     IDs_predictY = indId_predictY.copy()
#     IDs_predictY.update(orgId_predictY)
#     # print("length of IDs_predictY: ", len(IDs_predictY))
#     keys = list(IDs_predictY.keys())
#     # print(len(keys))
#
#     y_true = []
#     y_predic = []
#     for item in keys:
#         if item in indId_predictY.keys() not in orgId_predictY.keys():
#             y_true.append(int(0))
#             y_predic.append(indId_predictY[item])
#         elif item in orgId_predictY.keys() not in indId_predictY.keys():
#             y_true.append(int(1))
#             y_predic.append(orgId_predictY[item])
#
#     # print(len(y_true), len(y_predic))
#
#     with open(outfile, 'a') as f:
#         f.write(json.dumps(keys))
#         f.write('\n')
#         f.write(json.dumps(y_true))
#         f.write('\n')
#         f.write(json.dumps(y_predic))
#         f.write('\n')

if __name__ == '__main__':
    screenNames_org = load_screenNames1("./org.txt")
    screenNames_ind = load_screenNames2("./Indi.txt")
    get_history_tweets(screenNames_org, "./history_tweets_org.txt")
    get_history_tweets(screenNames_ind, "./history_tweets_ind.txt")



