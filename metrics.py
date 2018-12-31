import numpy as np

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
        accuracy = round(accuracy*100, 2)
        self.accu = accuracy
        return accuracy

    def error(self, conf_matrix):
        diagonals_sum = np.trace(conf_matrix)
        total = sum(conf_matrix.sum(axis = 1))
        error = (1 - (diagonals_sum/total))
        error = round(error, 4)
        return error

    def precision(self, conf_matrix):
        precisions = []

        for i in range(conf_matrix.shape[0]):
            tp = conf_matrix[i, i]
            tpfn = sum(np.sum(conf_matrix, axis=0)[i])
            precision = tp/tpfn
            precisions.append(precision)

        average_precision = sum(precisions)/len(precisions)

        self.preci = average_precision
        return round(average_precision, 3)
    
    def recall(self, conf_matrix):
        recalls = []

        for i in range(conf_matrix.shape[0]):
            tp = conf_matrix[i, i]
            tpfp = sum(np.sum(conf_matrix, axis=1)[i])
            recall = tp/tpfp
            recalls.append(recall)

        average_recall = sum(recalls)/len(recalls)

        self.recal = average_recall
        return round(average_recall, 3)

    def f_score(self):
        f1_score = 2*((self.preci*self.recal)/(self.preci+self.recal))
        return f1_score
    