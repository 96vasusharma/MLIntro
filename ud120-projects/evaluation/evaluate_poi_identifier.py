#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 



print "-------------------------------"
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


clf = tree.DecisionTreeClassifier()
clf.fit(features, labels)

pred = clf.predict(features)

# print "Starting accuracy of overfit Decision Tree =", clf.score(features, labels)
print "Starting accuracy of overfit Decision Tree =", accuracy_score(labels, pred)
# ------------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
# ------------------------------------------------------------------------------------

clf.fit(features_train, labels_train)

pred = clf.predict(features_test)

no_of_poi = 0
for _ in pred:
    if _ == 1:
        no_of_poi += 1
print "No. of poi in test set =", no_of_poi
print "Total no. of people in test set =", len(pred)

print "New accuracy using train_test_split() =", accuracy_score(labels_test, pred)

# biased still better
print "Accuracy using all 0.0 for pred list =", accuracy_score(labels_test, [0.]*len(pred))


similarity = 0

for i,j in zip(pred, labels_test):
    if i == j:
        similarity += 1
    else:
        print "Predicted =", i, "& True Label =", j

if similarity:
    print "No True positives"
else:
    print "No. of True positives =", similarity


print "Precision score =", precision_score(pred, labels_test)
print "Recall score =", recall_score(labels_test, pred)

print "-------------------------------"
