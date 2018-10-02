# Step 3: Training and ROC for Linear SVM, Random forest classifier, Logistic regression classifier, and Decision Tree classifier
    # https://pdfs.semanticscholar.org/8db1/5d1ab276fd5460b6ecccb5354655aa6ee7bd.pdf
    # https://github.com/melifluos/bayesian-age-detection

# Without historical tweets

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics, svm
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def load(filename, L):
    temp = []
    lab = []
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

# Binarize the output
y = label_binarize(y, classes=[1, 2, 3, 4, 5, 6, 7])
n_classes = y.shape[1]

X = CountVectorizer().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
print "X_train.shape", X_train.shape, "  y_train.shape", y_train.shape
print "X_test.shape", X_test.shape, "  y_test.shape", y_test.shape

def plotting(Y_Test, Y_Score):

    FPR = dict()
    TPR = dict()
    Roc_Auc = dict()

    for i in range(n_classes):
        FPR[i], TPR[i], _ = roc_curve(Y_Test[:, i], Y_Score[:, i])
        Roc_Auc[i] = auc(FPR[i], TPR[i])

    # First aggregate all false positive rates
    All_FPR = np.unique(np.concatenate([FPR[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    Mean_TPR = np.zeros_like(All_FPR)
    for i in range(n_classes):
        Mean_TPR += np.interp(All_FPR, FPR[i], TPR[i])

    # Compute micro-average ROC curve and ROC area
    FPR["micro"], TPR["micro"], _ = roc_curve(Y_Test.ravel(), Y_Score.ravel())

    # Finally average it and compute AUC
    Mean_TPR /= n_classes
    FPR["macro"] = All_FPR
    TPR["macro"] = Mean_TPR

    Roc_Auc["micro"] = auc(FPR["micro"], TPR["micro"])
    Roc_Auc["macro"] = auc(FPR["macro"], TPR["macro"])

    return FPR, TPR, Roc_Auc

# ---------------------------------------------------------------------------------------------------------
# svc = LinearSVC(random_state=0, multi_class='ovr')

classifier = OneVsRestClassifier(svm.SVC(kernel='linear', random_state=0))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)
predicted_label = classifier.predict(X_test)
Acc = accuracy_score(y_test, predicted_label)
print "OneVsRest svm", Acc

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# roc_auc_svm = auc(fpr_svm, tpr_svm)
print("Area under ROC curve for SVM is: ", roc_auc)
color = ['pink', 'red', 'yellow', 'black', 'blue', 'navy', 'aqua']
plt.figure()
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], color=color[i], lw=2, label='SVM ROC curve (area = %0.2f)' % roc_auc[i])

fpr, tpr, roc_auc = plotting(y_test, y_score)
plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve for SVM (area = {0:0.2f})'
                                           ''.format(roc_auc["micro"]), color='pink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve for SVM (area = {0:0.2f})'
                                           ''.format(roc_auc["macro"]), color='red', linestyle=':', linewidth=4)

# ---------------------------------------------------------------------------------------------------------

clf = OneVsRestClassifier(DecisionTreeClassifier(random_state=0))
clf.fit(X_train, y_train)
y_score = clf.predict(X_test)
Acc_dt = accuracy_score(y_test, y_score)
print "DT Acc=", Acc_dt
# print "cross_val_score =", cross_val_score(clf, X, y, cv=10)

# Plot all ROC curves
fpr, tpr, roc_auc = plotting(y_test, y_score)
plt.plot(fpr["micro"], tpr["micro"],label='micro-average ROC curve for DT (area = {0:0.2f})'
                                          ''.format(roc_auc["micro"]), color='yellow', linestyle=':', linewidth=3)

plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve for DT (area = {0:0.2f})'
                                           ''.format(roc_auc["macro"]), color='orange', linestyle=':', linewidth=3)

# # ---------------------------------------------------------------------------------------------------------
#
# clf = OneVsRestClassifier(RandomForestClassifier(random_state=0, class_weight="balanced"))
# clf.fit(X_train, y_train)
# y_score = clf.predict(X_test)
# Acc_rf = accuracy_score(y_test, y_score)
# print "RF Acc=", Acc_rf
#
# fpr, tpr, roc_auc = plotting(y_test, y_score)
# plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve for RF (area = {0:0.2f})'
#                                            ''.format(roc_auc["micro"]), color='blue', linestyle=':', linewidth=3.5)
#
# plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve for RF (area = {0:0.2f})'
#                                            ''.format(roc_auc["macro"]), color='navy', linestyle=':', linewidth=3.5)
#
# # ---------------------------------------------------------------------------------------------------------
#
# clf = OneVsRestClassifier(LogisticRegressionCV(multi_class='ovr', cv=10))
# clf.fit(X_train, y_train)
# y_score = clf.predict(X_test)
# Acc_lr = accuracy_score(y_test, y_score)
# print "LR Acc=", Acc_lr
#
# fpr, tpr, roc_auc = plotting(y_test, y_score)
# plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve for LR (area = {0:0.2f})'
#                                            ''.format(roc_auc["micro"]), color='aqua', linestyle=':', linewidth=2.5)
#
# plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve for LR (area = {0:0.2f})'
#                                            ''.format(roc_auc["macro"]), color='cornflowerblue', linestyle=':', linewidth=2.5)
#
# ---------------------------------------------------------------------------------------------------------

plt.xlim([-0.02, 1.0])
plt.ylim([0.5, 1.02])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()
