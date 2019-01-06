import numpy as np
from preprocess import *
import random
from KNN import *

class Metrics:
    def confusion_matrix(self, y_test, y_pred):
        classes = []
        for item in y_test:
            if item in classes:
                continue
            else:
                classes.append(item)
        
        conf_matrix = np.array([])
        conf_matrix = np.zeros((len(classes), len(classes)), dtype=int)

        for item in range(len(y_test)):
            actual = y_test[item]
            pred = y_pred[item]

            row = classes.index(actual)
            col = classes.index(pred)

            conf_matrix[row,col] += 1
        self.conf_matrix = conf_matrix
        return conf_matrix

    def accuracy(self, conf_matrix):
        diagonals_sum = np.trace(conf_matrix)
        total = sum(conf_matrix.sum(axis = 1))
        accuracy = diagonals_sum/total
        accuracy = round(accuracy, 2)
        self.accu = accuracy
        return accuracy

    def error(self, conf_matrix):
        diagonals_sum = np.trace(conf_matrix)
        total = sum(conf_matrix.sum(axis = 1))
        error = (1 - (diagonals_sum/total))
        error = round(error, 3)
        return error

    def precision(self, conf_matrix):
        precisions = []

        for i in range(conf_matrix.shape[0]):
            tp = conf_matrix[i, i]
            tpfn = np.sum(conf_matrix, axis=0)[i]
            precision = tp/tpfn
            precisions.append(precision)

        average_precision = sum(precisions)/len(precisions)

        self.preci = average_precision
        return round(average_precision, 3)
    
    def recall(self, conf_matrix):
        recalls = []

        for i in range(conf_matrix.shape[0]):
            tp = conf_matrix[i, i]
            tpfp = np.sum(conf_matrix, axis=1)[i]
            recall = tp/tpfp
            recalls.append(recall)

        average_recall = sum(recalls)/len(recalls)

        self.recal = average_recall
        return round(average_recall, 3)

    #binary classifier
    def sensitivity(self, conf_matrix):
        tp = conf_matrix[0, 0]
        fn = conf_matrix[1, 0]
        sensi = tp/(tp+fn)
        return round(sensi, 3)

    #binary classifier
    def specificity(self, conf_matrix):
        tn = conf_matrix[1, 1]
        fp = conf_matrix[0, 1]
        specifi = tn/(tn+fp)
        return round(specifi, 3)

    def f_score(self):
        f1_score = 2*((self.preci*self.recal)/(self.preci+self.recal))
        return round(f1_score, 3)
    
    def k_foldcross_validation(self, data, label_pos, K, n_estimator, n=10):
        #shuffle data
        random.shuffle(data)

        each_set_num = int(len(data)/n)
        
        #for each iteration in n, pick one in sets as test data and combine the rest as training data
        preprocess = Preprocess()
        
        accuracies = []
        precisions = []
        recalls = []
        errors = []
        fs = []
        specificities = []

        for i in range(n):
            test = []

            if i == 0:
                start_index = 0
                end_index = each_set_num

            n = start_index
            data_copy = data.copy()

            for n in range(start_index, end_index):
                # n = start_index
                item = data_copy[n]
                test.append(item)
                del data_copy[n]

            train = data_copy
            start_index = end_index
            end_index = end_index + each_set_num

            #seperate feat and label
            X_train, y_train, X_test, y_test = preprocess.seprate_feat_label(train, test, label_pos)  
            X_train = np.array(preprocess.normalise_data(X_train))
            X_test = np.array(preprocess.normalise_data(X_test))         
       
            knn_ensemble = KNNEnsemble("bagging", K)
            knn_ensemble.fit(X_train, y_train)
            voted_pred = knn_ensemble.bagging(X_test, -1, bags=n_estimator)
            conf_matrix = self.confusion_matrix(y_test, voted_pred)
            accuracy = self.accuracy(conf_matrix)
            error = self.error(conf_matrix)
            precision = self.precision(conf_matrix)
            recall = self.recall(conf_matrix)
            fsc = self.f_score()
            specifici = self.specificity(conf_matrix)

            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            errors.append(error)
            fs.append(fsc)
            specificities.append(specifici)

        metrics = [accuracies, precisions, recalls, errors, fs, specificities]

        return metrics





