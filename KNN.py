import math
import numpy as np

class KNN:
    def __init__ (self, k):
        self.k = k

    def euclidean_distance(self, point1, point2):
        total = 0
        for i in range(len(point1)-1):
            total = total + pow(point1[i] + point2[i], 2)

        return math.sqrt(total)

    def k_neighbors(self, train_set, test_set):
        distances, neighbors = [], []

        #iterate till len(train_set) - 1 assuming the features are in the last column
        for i in range(len(train_set) - 1):
            distance = self.euclidean_distance(train_set[i], test_set)
            distances.append((train_set[i], distance))

        #sort the distances
        distances = sorted(distances, key=lambda tup: tup[1])

        #pick k neighbors
        for k in range(self.k):
            neighbors.append(distances[k])

        #iterate through neighbors and assign label
        votes = {}
        for item in neighbors:
            i = item[0][-1]
            if i in votes:
                votes[i] += 1
            else:
                votes[i] = 1
        
        highest_vote = 0
        label = None
        for key, value in votes.items():
            if value > highest_vote:
                highest_vote = value
                label = key
        
        return label

    def predict(self, train_set, test_set):

        for each in test_set:
            label = self.k_neighbors(train_set, each)
            each.append(label)
        
        return test_set


KNN = KNN(2)
trainSet = [[2, 2, 2, 'a'], [4, 4, 4, 'b'], [7, 8, 9, 'c']]
testInstance = [[5, 5, 5], [9, 7, 5]]
KNN.predict(trainSet, testInstance)