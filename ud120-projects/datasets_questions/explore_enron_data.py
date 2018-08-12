#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
count = 0
for name in enron_data:
    if enron_data[name]["poi"]==1:
        count+=1
print "Person of interest ",count

# print enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]

print "LAY KENNETH L money = ",enron_data["LAY KENNETH L"]["total_payments"]
print "SKILLING JEFFREY K money = ",enron_data["SKILLING JEFFREY K"]["total_payments"]
print "FASTOW ANDREW S money = ",enron_data["FASTOW ANDREW S"]["total_payments"]

count = 0
for name in enron_data:
    if enron_data[name]["salary"]!="NaN":
        count+=1
print "People having valid salary",count


count = 0
for name in enron_data:
    if enron_data[name]["email_address"]!="NaN":
        count+=1
print "Known email address",count

count = 0
for name in enron_data:
    if enron_data[name]["total_payments"]=="NaN":
        count+=1
print count,len(enron_data.keys())
print "% of people having NaN Total payments",((count*1.0)/len(enron_data.keys()))*100


count = 0
for name in enron_data:
    if enron_data[name]["poi"]==1 and enron_data[name]["total_payments"]=="NaN":
        count+=1
print "% of people of interest having NaN Total payments",((count*1.0)/18)*100