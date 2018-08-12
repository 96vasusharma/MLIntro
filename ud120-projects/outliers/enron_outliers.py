#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]

#removing outlier TOTAL
data_dict.pop("TOTAL",0)

data = featureFormat(data_dict, features)


### your code below

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter(salary,bonus)

matplotlib.pyplot.xlabel("Salary")
matplotlib.pyplot.ylabel("Bonus")
matplotlib.pyplot.show()

# outlier=0
# for point in data:
#     if point[0] > 2.5e+07:
#         print "Outlier having largest salary =",point[0],\
#             "and bonus =",point[1]
#         outlier = point[0]
#
# for name in data_dict:
#     if data_dict[name]["salary"] == outlier:
#         print "Outlier name is",name

print "Person having salary > 1M & bonus > 5M are \n"
for name in data_dict:
    salary = data_dict[name]["salary"]
    if salary !="NaN" and salary > 1000000:
        bonus = data_dict[name]["bonus"]
        if bonus !="NaN" and bonus > 5000000:
            print name

