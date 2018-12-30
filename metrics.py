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
        
        return conf_matrix