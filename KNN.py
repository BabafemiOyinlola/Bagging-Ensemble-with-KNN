import math
import random
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
        
class KNNEnsemble(KNN):
    def __init__(self, method, k):
        self.method = method
        self.k = k

    def fit(self, train_feature, train_label):
        self.train_feature = train_feature
        self.train_label = train_label

    def bagging(self, bags=10, split_ratio=0.6):
            train = []
            for i in self.train_feature:
                temp = self.train_feature[i]
                temp.append(self.train_label[i])
                train.append(temp)
            
            predictions = []
            #resample from the training set
            for i in range(bags):
                samp_num = int(split_ratio * len(train))
                sample = []
                for i in range(samp_num):
                    index = random.randint(0, (len(train)-1))
                    item = train[index]
                    sample.append(item)

                    X_train = sample[:len(sample[0])-1]   
                    y_train = sample[:-1]

                    #fit to knn classifier
                    knn = super().__init__(self.k)
                    knn.fit(X_train, y_train)
                    y_pred = knn.predict(X_test)
                    predictions.append(y_pred)
            
            voted_predictions = []
            #vote across the predictions
            for i in predictions:
                clases = {}
                for j in predictions[i]:
                    if j in clases:
                        clases[j] += 1
                    else:
                        clases[j] = 1
                
                highest_vote = 0
                label = None
                for key, value in clases.items():
                    if value > highest_vote:
                        highest_vote = value
                        label = key
                voted_predictions.append(label)

            return voted_predictions

    

            





KNN = KNN(6)
# X_train = [[7, 8, 9], [4, 4, 4],[2, 2, 2]]
# y_train = ['a', 'b', 'c']
# X_test = [[5, 5, 5], [9, 7, 5]]

X_train = [[5.1,3.5,1.4,0.2], [4.9,3.0,1.4,0.2],[4.7,3.2,1.3,0.2],
            [7.0,3.2,4.7,1.4],[6.4,3.2,4.5,1.5],[6.9,3.1,4.9,1.5],
            [6.3,3.3,6.0,2.5],[5.8,2.7,5.1,1.9],[7.1,3.0,5.9,2.1],
            [5.2,2.7,3.9,1.4], [4.9,3.0,1.4,0.2],[4.7,3.2,1.3,0.2]]
y_train = ['setosa', 'setosa', 'setosa', 'versicolor', 'versicolor', 'versicolor', 'virginica', 'virginica', 'virginica', 'versicolor', 'versicolor',
            'setosa']
X_test = [[5.2,2.7,3.9,1.4], [4.4,3.0,1.3,0.2]]

KNN.fit(X_train, y_train)
y_pred = KNN.predict(X_test)
print("Done")
