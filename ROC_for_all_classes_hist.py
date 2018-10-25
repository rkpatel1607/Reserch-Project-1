import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.svm import LinearSVC
from sklearn import metrics
import pickle
import matplotlib.pyplot as plt
from sklearn import metrics, svm
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV

stopwords = []
sw = open("stopwords_en.txt", "r")
for word in sw:
    stopwords.append(word.rstrip())

def stopwords_removal(current_twt):
    updated_twt = ' '
    current_twt = current_twt.lower()
    for twt_word in current_twt.encode("utf-8").split():
        if twt_word not in stopwords:
            updated_twt = updated_twt + twt_word + ' '
    return updated_twt

# def cal(f, L):
#     user_twt = []
#     label = []
#     for line in f:
#         user_tweets = json.loads(line)
#         for item in user_tweets.items():
#             hist_twt = ''
#             count = 0
#             for T_ID in item[1]:
#                 count += 1
#             if count > 10:
#                 for T_ID in item[1]:
#                     twt = item[1][T_ID][u'text']
#                     hist_twt = hist_twt + stopwords_removal(twt)
#                 user_twt.append(hist_twt)
#                 label.append(L)
#     return user_twt, label
#
# f1 = open("hist_twts/historical_tweets_group_13_18_tweet_n_date_info.txt", "r")
# f2 = open("hist_twts/historical_tweets_group_18_24_tweet_n_date_info.txt", "r")
# f3 = open("hist_twts/historical_tweets_group_25_34_tweet_n_date_info.txt", "r")
# f4 = open("hist_twts/historical_tweets_group_35_44_tweet_n_date_info.txt", "r")
# f5 = open("hist_twts/historical_tweets_output_45_54.txt", "r")
# f6 = open("hist_twts/historical_tweets_output_55_64.txt", "r")
# f7 = open("hist_twts/historical_tweets_group_64_plus_tweet_n_date_info.txt", "r")
#
# U1, L1 = cal(f1, 1)
# U2, L2 = cal(f2, 2)
# U3, L3 = cal(f3, 3)
# U4, L4 = cal(f4, 4)
# U5, L5 = cal(f5, 5)
# U6, L6 = cal(f6, 6)
# U7, L7 = cal(f7, 7)
#
# X = U1 + U2 + U3 + U4 + U5 + U6 + U7
# transformer = TfidfVectorizer(stop_words="english")
# X = transformer.fit_transform(X)
#
# y = L1 + L2 + L3 + L4 + L5 + L6 + L7
# y = label_binarize(y, classes=[1, 2, 3, 4, 5, 6, 7])
# n_classes = y.shape[1]

[X, y] = pickle.loads(open("Prof_Code/X_Y.pkl").read())

X_train, X_test, y_train, y_test = train_test_split(X, np.array(y), test_size=0.4)
print "X_train.shape", X_train.shape, "  y_train.shape", y_train.shape
print "X_test.shape", X_test.shape, "  y_test.shape", y_test.shape

# # -------------------------------------------------------------------------------------------------

# classifier = OneVsRestClassifier((svm.SVC(kernel='linear', random_state=0)))
# y_score = classifier.fit(X_train, y_train).decision_function(X_test)
# predicted_label = classifier.predict(X_test)
#
# Acc = accuracy_score(y_test, predicted_label)
# print "OneVsRest svm acc = ", Acc
#
# # Compute ROC curve and ROC area for each class
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])
#
# # First aggregate all false positive rates
# All_FPR = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
#
# # Then interpolate all ROC curves at this points
# Mean_TPR = np.zeros_like(All_FPR)
# for i in range(n_classes):
#     Mean_TPR += np.interp(All_FPR, fpr[i], tpr[i])
#
# # Compute micro-average ROC curve and ROC area
# fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#
# # Finally average it and compute AUC
# Mean_TPR /= n_classes
# fpr["macro"] = All_FPR
# tpr["macro"] = Mean_TPR
# roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
#
# color = ['pink', 'red', 'yellow', 'black', 'blue', 'navy', 'aqua']
# plt.figure()
#
# plt.plot(fpr["micro"], tpr["micro"], label='micro-avg ROC, SVC (area = {0:0.2f})'.format(roc_auc["micro"]), color='blue', linestyle=':', linewidth=3.5)
# plt.plot(fpr["macro"], tpr["macro"], label='macro-avg ROC, SVC (area = {0:0.2f})'.format(roc_auc["macro"]), color='navy', linestyle=':', linewidth=3.5)
#
# for i in range(n_classes):
#     plt.plot(fpr[i], tpr[i], color=color[i], lw=2, label='ROC for class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
#
# plt.plot([0, 1], [0, 1], 'k--', lw=2)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.02])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.legend(loc="lower right")
# plt.show()
#

# # -------------------------------------------------------------------------------------------------

# clf = OneVsRestClassifier(DecisionTreeClassifier(random_state=0))
# clf.fit(X_train, y_train)
# y_score = clf.predict(X_test)
# Acc_dt = accuracy_score(y_test, y_score)
# print "DT Acc=", Acc_dt
#
# # Compute ROC curve and ROC area for each class
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])
#
# # First aggregate all false positive rates
# All_FPR = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
#
# # Then interpolate all ROC curves at this points
# Mean_TPR = np.zeros_like(All_FPR)
# for i in range(n_classes):
#     Mean_TPR += np.interp(All_FPR, fpr[i], tpr[i])
#
# # Compute micro-average ROC curve and ROC area
# fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#
# # Finally average it and compute AUC
# Mean_TPR /= n_classes
# fpr["macro"] = All_FPR
# tpr["macro"] = Mean_TPR
# roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
#
# color = ['pink', 'red', 'yellow', 'black', 'blue', 'navy', 'aqua']
# plt.figure()
# plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve for DT (area = {0:0.2f})'
#                                            ''.format(roc_auc["micro"]), color='blue', linestyle=':', linewidth=3.5)
#
# plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve for DT (area = {0:0.2f})'
#                                            ''.format(roc_auc["macro"]), color='navy', linestyle=':', linewidth=3.5)
# for i in range(n_classes):
#     plt.plot(fpr[i], tpr[i], color=color[i], lw=2, label='ROC for class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
#
# plt.xlim([-0.02, 1.0])
# plt.ylim([0.0, 1.02])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.legend(loc="lower right")
# plt.show()

# # ---------------------------------------------------------------------------------------------------------

clf = OneVsRestClassifier(RandomForestClassifier(random_state=0, class_weight="balanced"))
clf.fit(X_train, y_train)
y_score = clf.predict(X_test)
Acc_rf = accuracy_score(y_test, y_score)
# print "RF Acc=", Acc_rf

y_test = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6])
n_classes = y_test.shape[1]
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# First aggregate all false positive rates
All_FPR = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
Mean_TPR = np.zeros_like(All_FPR)
for i in range(n_classes):
    Mean_TPR += np.interp(All_FPR, fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Finally average it and compute AUC
Mean_TPR /= n_classes
fpr["macro"] = All_FPR
tpr["macro"] = Mean_TPR
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

color = ['pink', 'red', 'yellow', 'black', 'blue', 'navy', 'aqua']
plt.figure()
plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve for RF (area = {0:0.2f})'
                                           ''.format(roc_auc["micro"]), color='blue', linestyle=':', linewidth=3.5)

plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve for RF (area = {0:0.2f})'
                                            ''.format(roc_auc["macro"]), color='navy', linestyle=':', linewidth=3.5)

for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], color=color[i], lw=2, label='ROC for class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.02])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

# # -------------------------------------------------------------------------------------------------

# clf = OneVsRestClassifier(LogisticRegressionCV(multi_class='ovr', cv=10))
# clf.fit(X_train, y_train)
# y_score = clf.predict(X_test)
# Acc_lr = accuracy_score(y_test, y_score)
# print "LR Acc=", Acc_lr
#
# # Compute ROC curve and ROC area for each class
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])
#
# # First aggregate all false positive rates
# All_FPR = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
#
# # Then interpolate all ROC curves at this points
# Mean_TPR = np.zeros_like(All_FPR)
# for i in range(n_classes):
#     Mean_TPR += np.interp(All_FPR, fpr[i], tpr[i])
#
# # Compute micro-average ROC curve and ROC area
# fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#
# # Finally average it and compute AUC
# Mean_TPR /= n_classes
# fpr["macro"] = All_FPR
# tpr["macro"] = Mean_TPR
# roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
#
# color = ['pink', 'red', 'yellow', 'black', 'blue', 'navy', 'aqua']
# plt.figure()
#
# plt.plot(fpr["micro"], tpr["micro"], label='micro-avg ROC, LR (area = {0:0.2f})'.format(roc_auc["micro"]), color='blue', linestyle=':', linewidth=3.5)
# plt.plot(fpr["macro"], tpr["macro"], label='macro-avg ROC, LR (area = {0:0.2f})'.format(roc_auc["macro"]), color='navy', linestyle=':', linewidth=3.5)
#
# for i in range(n_classes):
#     plt.plot(fpr[i], tpr[i], color=color[i], lw=2, label='ROC for class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
#
# plt.plot([0, 1], [0, 1], 'k--', lw=2)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.02])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.legend(loc="lower right")
# plt.show()


