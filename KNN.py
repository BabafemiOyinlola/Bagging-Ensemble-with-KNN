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

    def fit(self, train_feature, train_label):
        self.train_feature = train_feature
        self.train_label = train_label

    def k_neighbors(self, train_set, test_set):
        distances, neighbors = [], []
        #iterate till len(train_set) - 1 assuming the features are in the last column
        for i in range(len(train_set)):
            distance = self.euclidean_distance(train_set[i], test_set)
            train_set[i].append(self.train_label[i])
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

    def predict(self, test_feature):
        # train_set = np.column_stack([self.train_feature, self.train_label]).astype(np.float)
        test_feature = test_feature
        predicted_label = []
        for each in test_feature:
            label = self.k_neighbors(self.train_feature, each)
            each.append(label)
            predicted_label.append(label)
        
        return predicted_label

    def confussion_matrix(self,true_label, pred_label):
        true_classes, pred_classes = {}, {}
        for label in true_label:
            if label in true_classes:
                true_classes[label] += 1
            else:
                true_classes[label] == 1

        for label in pred_label:
            if label in pred_classes:
                pred_classes[label] += 1
            else:
                pred_classes[label] == 1
        

KNN = KNN(2)
X_train = [[7, 8, 9], [4, 4, 4],[2, 2, 2]]
y_train = ['a', 'b', 'c']
X_test = [[5, 5, 5], [9, 7, 5]]

KNN.fit(X_train, y_train)
y_pred = KNN.predict(X_test)

trainSet = [[2, 2, 2, 'a'], [4, 4, 4, 'b'], [7, 8, 9, 'c']]
