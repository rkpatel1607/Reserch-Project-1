# Step 2: Train a linear SVM classifier for the set of 700 users | 7-class SVM classifier
#   http://scikit-learn.org/stable/modules/multiclass.html

import json
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import label_binarize
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
# vectorizer = CountVectorizer(lowercase=False, stop_words='english')

def load(filename, L):
    temp = [];    lab = []
    f = open(filename)
    c = 0
    for line in f:
        if c < 100:
            c += 1
            tmp = json.loads(line)
            temp.append(tmp['tweet'].encode('utf-8'))
            lab.append(L)
        else:
            break
    return temp, lab

data1, L1 = load("Twt_n_date_info/group_13_18_tweet_n_date_info.txt", 1)
data2, L2 = load("Twt_n_date_info/group_18_24_tweet_n_date_info.txt", 2)
data3, L3 = load("Twt_n_date_info/group_25_34_tweet_n_date_info.txt", 3)
data4, L4 = load("Twt_n_date_info/group_35_44_tweet_n_date_info.txt", 4)
data5, L5 = load("Twt_n_date_info/group_45_54_tweet_n_date_info.txt", 5)
data6, L6 = load("Twt_n_date_info/group_55_64_tweet_n_date_info.txt", 6)
data7, L7 = load("Twt_n_date_info/group_64_plus_tweet_n_date_info.txt", 7)

X = data1 + data2 + data3 + data4 + data5 + data6 + data7
y = L1 + L2 + L3 + L4 + L5 + L6 + L7
# print X, "\n", y

X = CountVectorizer().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, np.array(y), test_size=0.4)
print "X_train.shape", X_train.shape, "  y_train.shape", y_train.shape
print "X_test.shape", X_test.shape, "  y_test.shape", y_test.shape

svc = LinearSVC(random_state=0, multi_class='ovr')

SVM = svc.fit(X_train, y_train)
predicted_label = svc.predict(X_test)
Acc_no_fold = accuracy_score(y_test, predicted_label)
print "\n Accuracy_without_CV: ", Acc_no_fold

predicted = cross_val_predict(SVM, X, y, cv=10)
Acc_fold = metrics.accuracy_score(y, predicted)
print "\n Accuracy_with_10_fold_CV: ", Acc_fold
