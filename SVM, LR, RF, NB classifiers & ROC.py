# task 2: Train SVM, LR, RF, and NB classifiers and generate ROC curves for the four classifiers.

import numpy as np
import pandas as pd
from TwitterAPI import TwitterAPI
from datetime import datetime

from sklearn import svm
from sklearn.feature_extraction.text import *
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import auc, roc_curve

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

t1 = datetime.now()
api = TwitterAPI('', '', '', '')


def load_data(filename):
    data = []
    op = open(filename, "r")
    for txt in op:
        data.append(txt.rstrip())
    return data


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


stopwords = load_data("stopwords_en.txt")
male_keywords = load_data("Male_Frequent_words.txt")
female_keywords = load_data("Female_Frequent_words.txt")
user_names = load_data("user_names.txt")

count = 0
tweets = []
cls_label = []
for name in user_names:
    if count < 450:
        count += 1
        try:
            results = api.request('statuses/user_timeline', {'screen_name': name, 'count': 200})
            User_tweets = ' '
            for result in results:
                User_tweets = User_tweets + stopwords_removal(result['text']) + ', '

            if len(User_tweets) != 0:
                tweets.append(User_tweets)
                male_count, female_count = identification(User_tweets)
                if male_count > female_count:
                    cls_label.append(1)
                else:
                    cls_label.append(0)
        except:
            print name
    else:
        break

X = np.array(tweets)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X).toarray()

X = pd.DataFrame(X)
X_train, X_test, y_train, y_test = train_test_split(X, np.array(cls_label), test_size=0.3)
print "X_train.shape", X_train.shape, "  y_train.shape", y_train.shape
print "X_test.shape", X_test.shape, "  y_test.shape", y_test.shape


# SVM
svc = svm.SVC(kernel='linear', probability=True, class_weight={1: 2})
Cs = range(1, 20)
SVM = GridSearchCV(estimator=svc, param_grid=dict(C=Cs), cv=10)
SVM_y = svc.fit(X_train, y_train).decision_function(X_test)

fpr_svm, tpr_svm, _ = roc_curve(y_test, SVM_y)
roc_auc_svm = auc(fpr_svm, tpr_svm)
print("Area under ROC curve for SVM is: ", roc_auc_svm)
plt.figure()
plt.plot(fpr_svm, tpr_svm, color='red', lw=2, label='SVM ROC curve (area = %0.2f)' % roc_auc_svm)


# LR
lr = LogisticRegression()
LR_y = lr.fit(X_train, y_train).decision_function(X_test)

fpr_lr, tpr_lr, _ = roc_curve(y_test, LR_y)
roc_auc_lr = auc(fpr_lr, tpr_lr)
print("Area under ROC curve for LR is: ", roc_auc_lr)
plt.plot(fpr_lr, tpr_lr, color='blue', lw=2, label='LR ROC curve (area = %0.2f)' % roc_auc_lr)


# RF
rf = RandomForestClassifier(max_depth=100, random_state=0)
RF_y = rf.fit(X_train, y_train).predict(X_test)

fpr_rf, tpr_rf, _ = roc_curve(y_test, RF_y)
roc_auc_rf = auc(fpr_rf, tpr_rf)
print("Area under ROC curve for RF is: ", roc_auc_rf)
plt.plot(fpr_rf, tpr_rf, color='yellow', lw=2, label='RF ROC curve (area = %0.2f)' % roc_auc_rf)


# NB
nb = MultinomialNB()
NB_y = nb.fit(X_train, y_train).predict(X_test)

fpr_nb, tpr_nb, _ = roc_curve(y_test, NB_y)
roc_auc_nb = auc(fpr_nb, tpr_nb)
print("Area under ROC curve for NB is: ", roc_auc_nb)
plt.plot(fpr_nb, tpr_nb, color='green', lw=2, label='NB ROC curve (area = %0.2f)' % roc_auc_nb)


# Plotting
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc="lower right")
plt.show()
plt.savefig('ROC.png')

t2 = datetime.now()
print "\n Run time: ", t2 - t1
