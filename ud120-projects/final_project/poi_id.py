#!/usr/bin/python

import sys
import matplotlib.pyplot
import numpy
import pickle
sys.path.append("../tools/")

from sklearn import tree
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.feature_extraction.text import TfidfVectorizer

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi', 'salary','bonus', 'long_term_incentive',
                 'total_payments', 'expenses','other','exercised_stock_options',
                 'director_fees','restricted_stock', 'from_poi_to_this_person',
                 'to_messages', 'from_messages', 'from_this_person_to_poi',
                 'shared_receipt_with_poi']  # You will need to use more features

# 'deferral_payments','loan_advances', 'restricted_stock_deferred',
# 'deferred_income', 'total_stock_value',


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.



my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


from sklearn.preprocessing import MinMaxScaler

# weights = numpy.array([[115],[140],[175]])  # make the val as float
scaler = MinMaxScaler()
rescaled_features = scaler.fit_transform(features)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html




# Provided to give you a starting point. Try a variety of classifiers.
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()


# GridSearchCV  (Auto tuning ) Decision tree
# parameters = {'criterion': ('gini', 'entropy'),
#               'splitter': ('best', 'random'),
#               'min_samples_split': range(2, 10)}
#
# dtc = tree.DecisionTreeClassifier()
# clf = GridSearchCV(dtc, parameters)


estimators = [('reduce_dim', PCA(n_components=2)),
              ('clf', tree.DecisionTreeClassifier(min_samples_split=7,
                                                  splitter='random',
                                                  criterion='gini'))]
clf = Pipeline(estimators)

# Svm auto tune
# parameters = {'kernel': ('linear', 'poly', 'rbf','sigmoid'),
#               'C': range(1, 100)}
# svc = svm.SVC()
# clf = GridSearchCV(svc, parameters)

# -----------------------------------------------------
# SVM algo best prams :-
# clf = svm.SVC(C=43,kernel='sigmoid')

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(rescaled_features, labels, test_size=0.3, random_state=42)


# vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
#                                  stop_words='english')
# features_train_transformed = vectorizer.fit_transform(features_train)
# features_test_transformed = vectorizer.transform(features_test)


selector = SelectKBest(chi2, k=2)
selector.fit(features_train, labels_train)
features_train_transformed = selector.transform(features_train)
features_test_transformed = selector.transform(features_test)
print selector.scores_


# selector = SelectPercentile(f_classif, percentile=51)
# selector.fit(features_train, labels_train)
# features_train_transformed = selector.transform(features_train)
# features_test_transformed = selector.transform(features_test)



# from sklearn import linear_model
# clf = linear_model.Lasso(alpha=0.1)
# clf.fit(features_train, labels_train)
# print "Lasso coeff. 0 means discard it", sorted(clf.coef_)

clf.fit(features_train_transformed, labels_train)

# print clf.best_params_
# print clf.best_score_

pred = clf.predict(features_test_transformed)


no_of_poi = 0
for _ in pred:
    if _ == 1:
        no_of_poi += 1
print "No. of poi in test set =", no_of_poi
print "Total no. of people in test set =", len(pred)

print "New accuracy using train_test_split() =", accuracy_score(labels_test, pred)

# biased still better
print "Accuracy using all 0.0 for pred list =", accuracy_score(labels_test,
                                                               [0.]*len(pred))


similarity = 0

for i, j in zip(pred, labels_test):
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
print "F1 score =", f1_score(labels_test, pred)


print "Accuracy of Decision Tree =", accuracy_score(labels_test, pred)
# print clf.feature_importances_


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
