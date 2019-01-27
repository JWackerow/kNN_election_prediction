# -*- coding: utf-8 -*-
"""
k-nearest neighbors algorithm is a simple way to classify data. 
Here we use county data to predict how a county will vote in the 2016 election

@author: JWackerow
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import math
import operator

data = pd.read_csv("Election_County_Data.csv")
# split our data into training and test data
train, test = train_test_split(data, test_size=0.2)
train = train.values.tolist()
test = test.values.tolist()

# returns the euclidean distance of each of the counties
def euclidean_distance(test_instance, train_instance):
    distance = 0
    length = len(test_instance)
    for x in range(length):
        distance += pow(train_instance[x] - test_instance[x],2)
    return math.sqrt(distance)

# returns a list of counties with data closest to the test county           
def get_neighbors(test_instance, training_data, k):
    distances = []
    for county in training_data:
        distances.append((euclidean_distance(test_instance[1:-2], county[1:-2]), county[-1]))
    # sort and return counties with the smallest distances
    distances.sort(key=lambda tup: tup[0])
    return distances[:k]

# adds up the number of times a label/class appears in the county's list of neighbors and returns the label with the most votes
def get_label (neighbors):
    label_votes = {}
    for county, label in neighbors:
        if label not in label_votes:
            label_votes[label] = 1
        else:
            label_votes[label] += 1
    return sorted(label_votes.items(), key=operator.itemgetter(1), reverse = True)[0][0]

# prints the accuracy
def get_accuracy(k):
    correct = 0
    for county in test:
        neighbors = get_neighbors(county, train, k)
        if county[-1] == get_label(neighbors):
            correct += 1
    print("Accuracy: ", round((correct/len(test)) * 100, 1))

get_accuracy(3)