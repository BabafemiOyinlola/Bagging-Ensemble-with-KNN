from KNN import *
from metrics import *
from preprocess import *

preprocess = Preprocess()
# data = read_csv("/Users/oyinlola/Desktop/MSc Data Science/SCC403 - Data Mining/iris.csv", True)
data = preprocess.read_csv("/Users/oyinlola/Desktop/MSc Data Science/SCC461 - Programming for Data Scientists/Final Project/iris_data2.txt", False)
X_train, y_train, X_test, y_test = preprocess.split_data(data, -1, 0.2)

metrics = Metrics()

knn = KNN(8)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
conf_matrix = metrics.confusion_matrix(y_test, y_pred)

knn_ensemble = KNNEnsemble("bagging", 4)
knn_ensemble.fit(X_train, y_train)
voted_pred = knn_ensemble.bagging(X_test, -1)
conf_matrix = metrics.confusion_matrix(y_test, voted_pred)


print("Done")
