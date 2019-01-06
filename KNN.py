import math
import random
import numpy as np

class KNN:
    def __init__ (self, k):
        self.k = k
        self.train_feature = []
        self.train_label = []

    def euclidean_distance(self, point1, point2):
        total = 0
        for i in range(len(point1)-1):
            total = total + pow((point1[i] - point2[i]), 2)

        return math.sqrt(total)

    def fit(self, train_feature, train_label):
        self.train_feature = train_feature
        self.train_label = train_label

    def k_neighbors(self, test_set):
        distances, neighbors = [], []
        for i in range(len(self.train_feature)):
            distance = self.euclidean_distance(self.train_feature[i], test_set)
            tmp = list(self.train_feature[i].copy())
            tmp.append(self.train_label[i])
            distances.append((tmp, distance))

        #sort the distances from closest to farthest     
        distances = sorted(distances, key=lambda tup: tup[1])

        #pick k neighbors
        for k in range(self.k):
            neighbors.append(distances[k])

        #iterate through neighbors and assign label
        votes = {}
        for i in range(len(neighbors)):
            clas = neighbors[i][0][-1]
            if clas in votes:
                votes[clas] += 1
            else:
                votes[clas] = 1
        
        highest_vote = 0
        label = None
        for key, value in votes.items():
            if value > highest_vote:
                highest_vote = value
                label = key
        
        return label

    def predict(self, test_feature):
        predicted_label = []
        for i in range(len(test_feature)):
            label = self.k_neighbors(test_feature[i])
            predicted_label.append(label)
        
        return predicted_label
   
class KNNEnsemble(KNN):
    def __init__(self, method, k):
        self.method = method
        self.k = k

    def fit(self, train_feature, train_label):
        self.train_feature = train_feature
        self.train_label = train_label

    def bagging(self, X_test, label_pos, bags=10, split_ratio=0.6):
        '''
        label_pos gives the location of the label as either 0 or -1
        '''
        train = []

        #combine feature and label 
        temp = (self.train_feature.copy()).tolist()
        for i in range(len(temp)):
            temp[i].append(self.train_label[i])
            train.append(temp[i])
        
        predictions = []
        #resample from the training set
        for i in range(bags):
            samp_num = int(split_ratio * len(train))
            sample = []

            #pick sample with replacement
            for i in range(samp_num):
                index = random.randint(0, (len(train)-1))
                item = train[index]
                sample.append(item)

            #fit to classifer
            X_train, y_train = [], []

            if label_pos == -1:
                for i in range(len(sample)):
                    feat = sample[i]
                    feat = feat[0:len(sample[0])-1]
                    feat = [float(i) for i in feat]
                    X_train.append(feat)
                    y_train.append(sample[i][-1])

            elif label_pos == 0:
                for i in range(len(sample)):
                    feat = sample[i]
                    feat = feat[1:len(sample[0])]
                    feat = [float(i) for i in feat]
                    X_train.append(feat)
                    y_train.append(sample[i][0])

            X_train = np.array(X_train)
            #fit to knn classifier
            knn = KNN(self.k)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            predictions.append(y_pred)
        
        voted_predictions = []
        #vote across the predictions
        for i in range(len(predictions[0])):
            classes = []
            for j in range(len(predictions)):
                label = predictions[j][i]
                classes.append(label)
            
            vote = {}
            for k in classes:
                if k in vote:
                    vote[k] += 1
                else:
                    vote[k] = 1
            
            highest_vote = 0
            label = None
            for key, value in vote.items():
                if value > highest_vote:
                    highest_vote = value
                    label = key
            voted_predictions.append(label)

        return voted_predictions

        
    
    

            





# KNN = KNN(6)
# # X_train = [[7, 8, 9], [4, 4, 4],[2, 2, 2]]
# # y_train = ['a', 'b', 'c']
# # X_test = [[5, 5, 5], [9, 7, 5]]

# X_train = [[5.1,3.5,1.4,0.2], [4.9,3.0,1.4,0.2],[4.7,3.2,1.3,0.2],
#             [7.0,3.2,4.7,1.4],[6.4,3.2,4.5,1.5],[6.9,3.1,4.9,1.5],
#             [6.3,3.3,6.0,2.5],[5.8,2.7,5.1,1.9],[7.1,3.0,5.9,2.1],
#             [5.2,2.7,3.9,1.4], [4.9,3.0,1.4,0.2],[4.7,3.2,1.3,0.2]]
# y_train = ['setosa', 'setosa', 'setosa', 'versicolor', 'versicolor', 'versicolor', 'virginica', 'virginica', 'virginica', 'versicolor', 'versicolor',
#             'setosa']
# X_test = [[5.2,2.7,3.9,1.4], [4.4,3.0,1.3,0.2]]

# KNN.fit(X_train, y_train)
# y_pred = KNN.predict(X_test)
# print("Done")
