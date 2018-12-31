from KNN import *
from metrics import *
from preprocess import *
from using_scikit_learn import *

preprocess = Preprocess()
data = preprocess.read_csv("/Users/oyinlola/Desktop/MSc Data Science/SCC461 - Programming for Data Scientists/Final Project/iris_data2.txt", False)
# data = preprocess.read_csv("/Users/oyinlola/Desktop/MSc Data Science/SCC461 - Programming for Data Scientists/Final Project/wine_data.txt", False)
X_train, y_train, X_test, y_test = preprocess.split_data(data, -1, 0.3)

metrics = Metrics()

#library implementation
knn_lib = ScikitLearn()
lib_pred = knn_lib.KNNClassifier(10, X_train, y_train, X_test)
knn_lib.metrics_base(lib_pred, y_test)

knn_lib_ens = ScikitLearn()
lib_pred = knn_lib_ens.ensemble(10, X_train, y_train, X_test)
knn_lib_ens.metrics_ens(lib_pred, y_test)



#base classifier
knn = KNN(10)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
accuracy1 = metrics.accuracy(conf_matrix)
error1 = metrics.error(conf_matrix)

#ensemble
knn_ensemble = KNNEnsemble("bagging", 10)
knn_ensemble.fit(X_train, y_train)
voted_pred = knn_ensemble.bagging(X_test, -1)
conf_matrix2 = metrics.confusion_matrix(y_test, voted_pred)
accuracy2 = metrics.accuracy(conf_matrix2)
error2 = metrics.error(conf_matrix2)



print("Done")
