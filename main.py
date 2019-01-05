from KNN import *
from metrics import *
from preprocess import *
from using_scikit_learn import *


def write_to_csv(columns, names, filename):
    content = []
    line = ""
    for i in range(len(names)):
        line += names[i] + " , "
    content.append(line)     
    
    for i in range(len(columns[0])):
        vals = ""
        for j in range(len(columns)):
            vals += str(columns[j][i]) + " , "
        content.append(vals)
    
    filename = filename+'.csv'

    with open(filename,'wb') as file:
        for l in content:
            file.write(l.encode())
            file.write('\n'.encode())
    

preprocess = Preprocess()
ionosphere_data = preprocess.read_csv("/Users/oyinlola/Desktop/MSc Data Science/SCC461 - Programming for Data Scientists/Final Project/ionosphere.data.csv", False)
breast_cancer_data = preprocess.read_breast_cancer_data("/Users/oyinlola/Desktop/MSc Data Science/SCC461 - Programming for Data Scientists/Final Project/breast-cancer-wisconsin.csv", False)

X_train, y_train, X_test, y_test = preprocess.split_data(ionosphere_data, -1, 0.2)
X_train = np.array(preprocess.normalise_data(X_train))
X_test = np.array(preprocess.normalise_data(X_test))

metrics = Metrics()


#ensemble of base classifier
accuracies_ens = []
precisions_ens = []
recalls_ens = []
errors_ens = []
fs_ens = []
specificity_ens = []

#change k
for i in range(1, 21):
    knn_ensemble = KNNEnsemble("bagging", i)
    knn_ensemble.fit(X_train, y_train)
    voted_pred = knn_ensemble.bagging(X_test, -1, bags=5)
    conf_matrix2 = metrics.confusion_matrix(y_test, voted_pred)
    accuracy2 = metrics.accuracy(conf_matrix2)
    error2 = metrics.error(conf_matrix2)
    precision2 = metrics.precision(conf_matrix2)
    recall2 = metrics.recall(conf_matrix2)
    fs = metrics.f_score()
    specificity2 = metrics.specificity(conf_matrix2)

    accuracies_ens.append(accuracy2)
    precisions_ens.append(precision2)
    recalls_ens.append(recall2)
    errors_ens.append(error2)
    fs_ens.append(fs)
    specificity_ens.append(specificity2)

metrics_obtained = [accuracies_ens, precisions_ens, recalls_ens, errors_ens, fs_ens, specificity_ens]
write_to_csv(metrics_obtained, ["Accuracy", "Precision", "Recall", "Error", "F-score", "Specificity"], "Ionosphere-Scratch-Different-Ks")

# lib ens classifier
accuracies = []
precisions = []
recalls = []
errors = []
fs = []
specificity = []

for i in range(1, 21):
    knn_lib_ens = ScikitLearn()
    lib_pred = knn_lib_ens.ensemble(i, X_train, y_train, X_test, bags=5)
    metrics_ens = knn_lib_ens.metrics_ens(lib_pred, y_test)

    accuracies.append(metrics_ens[0])
    precisions.append(round(metrics_ens[2][0],3))
    recalls.append(round(metrics_ens[2][1], 3))
    errors.append(round(1-metrics_ens[0], 3))
    fs.append(round(metrics_ens[2][2], 3))

    conf_mat = metrics_ens[1]
    specif = conf_mat[1, 1]/(conf_mat[1, 1] + conf_mat[0, 1])

    specificity.append(round(specif, 3))

lib_metrics_obtained = [accuracies, precisions, recalls, errors, fs, specificity]
write_to_csv(lib_metrics_obtained, ["Accuracy", "Precision", "Recall", "Error", "F-score", "Specificity"], "Ionosphere-Lib-Different-Ks")

#=========================================================================================================================#
#=========================================================================================================================#
#=========================================================================================================================#
#=========================================================================================================================#

X_train, y_train, X_test, y_test = preprocess.split_data(breast_cancer_data, 0, 0.2)
X_train = np.array(preprocess.normalise_data(X_train))
X_test = np.array(preprocess.normalise_data(X_test))

#ensemble of base classifier
accuracies_ens = []
precisions_ens = []
recalls_ens = []
errors_ens = []
fs_ens = []
specificity_ens = []

#change k
for i in range(1, 21):
    knn_ensemble = KNNEnsemble("bagging", i)
    knn_ensemble.fit(X_train, y_train)
    voted_pred = knn_ensemble.bagging(X_test, -1, bags=5)
    conf_matrix2 = metrics.confusion_matrix(y_test, voted_pred)
    accuracy2 = metrics.accuracy(conf_matrix2)
    error2 = metrics.error(conf_matrix2)
    precision2 = metrics.precision(conf_matrix2)
    recall2 = metrics.recall(conf_matrix2)
    fs = metrics.f_score()
    specificity2 = metrics.specificity(conf_matrix2)

    accuracies_ens.append(accuracy2)
    precisions_ens.append(precision2)
    recalls_ens.append(recall2)
    errors_ens.append(error2)
    fs_ens.append(fs)
    specificity_ens.append(specificity2)

metrics_obtained = [accuracies_ens, precisions_ens, recalls_ens, errors_ens, fs_ens, specificity_ens]
write_to_csv(metrics_obtained, ["Accuracy", "Precision", "Recall", "Error", "F-score", "Specificity"], "Breast-Cancer-Scratch-Different-Ks")

# lib ens classifier
accuracies = []
precisions = []
recalls = []
errors = []
fs = []
specificity = []

for i in range(1, 21):
    knn_lib_ens = ScikitLearn()
    lib_pred = knn_lib_ens.ensemble(i, X_train, y_train, X_test, bags=5)
    metrics_ens = knn_lib_ens.metrics_ens(lib_pred, y_test)

    accuracies.append(metrics_ens[0])
    precisions.append(round(metrics_ens[2][0],3))
    recalls.append(round(metrics_ens[2][1], 3))
    errors.append(round(1-metrics_ens[0], 3))
    fs.append(round(metrics_ens[2][2], 3))

    conf_mat = metrics_ens[1]
    specif = conf_mat[1, 1]/(conf_mat[1, 1] + conf_mat[0, 1])

    specificity.append(round(specif, 3))

lib_metrics_obtained = [accuracies, precisions, recalls, errors, fs, specificity]
write_to_csv(lib_metrics_obtained, ["Accuracy", "Precision", "Recall", "Error", "F-score", "Specificity"], "Breast-Cancer-Lib-Different-Ks")

#=========================================================================================================================#
#=========================================================================================================================#
#=========================================================================================================================#
#=========================================================================================================================#
ionosphere_data = preprocess.read_csv("/Users/oyinlola/Desktop/MSc Data Science/SCC461 - Programming for Data Scientists/Final Project/ionosphere.data.csv", False)
breast_cancer_data = preprocess.read_breast_cancer_data("/Users/oyinlola/Desktop/MSc Data Science/SCC461 - Programming for Data Scientists/Final Project/breast-cancer-wisconsin.csv", False)


X_train, y_train, X_test, y_test = preprocess.split_data(ionosphere_data, -1, 0.2)
X_train = np.array(preprocess.normalise_data(X_train))
X_test = np.array(preprocess.normalise_data(X_test))



#ensemble of base classifier
accuracies_ens = []
precisions_ens = []
recalls_ens = []
errors_ens = []
fs_ens = []
specificity_ens = []

#change no of predictors at K = 2
for i in range(2, 21):
    knn_ensemble = KNNEnsemble("bagging", 2)
    knn_ensemble.fit(X_train, y_train)
    voted_pred = knn_ensemble.bagging(X_test, -1, bags=i)
    conf_matrix2 = metrics.confusion_matrix(y_test, voted_pred)
    accuracy2 = metrics.accuracy(conf_matrix2)
    error2 = metrics.error(conf_matrix2)
    precision2 = metrics.precision(conf_matrix2)
    recall2 = metrics.recall(conf_matrix2)
    fs = metrics.f_score()
    specificity2 = metrics.specificity(conf_matrix2)

    accuracies_ens.append(accuracy2)
    precisions_ens.append(precision2)
    recalls_ens.append(recall2)
    errors_ens.append(error2)
    fs_ens.append(fs)
    specificity_ens.append(specificity2)

metrics_obtained = [accuracies_ens, precisions_ens, recalls_ens, errors_ens, fs_ens, specificity_ens]
write_to_csv(metrics_obtained, ["Accuracy", "Precision", "Recall", "Error", "F-score", "Specificity"], "Ionosphere-Scratch-K=2")

# lib ens classifier
accuracies = []
precisions = []
recalls = []
errors = []
fs = []
specificity = []

#change no of predictors at K = 2
for i in range(2, 21):
    knn_lib_ens = ScikitLearn()
    lib_pred = knn_lib_ens.ensemble(2, X_train, y_train, X_test, bags=i)
    metrics_ens = knn_lib_ens.metrics_ens(lib_pred, y_test)

    accuracies.append(metrics_ens[0])
    precisions.append(round(metrics_ens[2][0],3))
    recalls.append(round(metrics_ens[2][1], 3))
    errors.append(round(1-metrics_ens[0], 3))
    fs.append(round(metrics_ens[2][2], 3))

    conf_mat = metrics_ens[1]
    specif = conf_mat[1, 1]/(conf_mat[1, 1] + conf_mat[0, 1])

    specificity.append(round(specif, 3))

lib_metrics_obtained = [accuracies, precisions, recalls, errors, fs, specificity]
write_to_csv(lib_metrics_obtained, ["Accuracy", "Precision", "Recall", "Error", "F-score", "Specificity"], "Ionosphere-Lib-K=2")

#=========================================================================================================================#
#=========================================================================================================================#
#=========================================================================================================================#
#=========================================================================================================================#

X_train, y_train, X_test, y_test = preprocess.split_data(breast_cancer_data, 0, 0.2)
X_train = np.array(preprocess.normalise_data(X_train))
X_test = np.array(preprocess.normalise_data(X_test))

#ensemble of base classifier
accuracies_ens = []
precisions_ens = []
recalls_ens = []
errors_ens = []
fs_ens = []
specificity_ens = []

#change no of predictors at K = 12
for i in range(2, 21):
    knn_ensemble = KNNEnsemble("bagging", 12)
    knn_ensemble.fit(X_train, y_train)
    voted_pred = knn_ensemble.bagging(X_test, -1, bags=i)
    conf_matrix2 = metrics.confusion_matrix(y_test, voted_pred)
    accuracy2 = metrics.accuracy(conf_matrix2)
    error2 = metrics.error(conf_matrix2)
    precision2 = metrics.precision(conf_matrix2)
    recall2 = metrics.recall(conf_matrix2)
    fs = metrics.f_score()
    specificity2 = metrics.specificity(conf_matrix2)

    accuracies_ens.append(accuracy2)
    precisions_ens.append(precision2)
    recalls_ens.append(recall2)
    errors_ens.append(error2)
    fs_ens.append(fs)
    specificity_ens.append(specificity2)

metrics_obtained = [accuracies_ens, precisions_ens, recalls_ens, errors_ens, fs_ens, specificity_ens]
write_to_csv(metrics_obtained, ["Accuracy", "Precision", "Recall", "Error", "F-score", "Specificity"], "Breast-Cancer-Scratch-K=12")

# lib ens classifier
accuracies = []
precisions = []
recalls = []
errors = []
fs = []
specificity = []

#change no of predictors at K = 6
for i in range(2, 21):
    knn_lib_ens = ScikitLearn()
    lib_pred = knn_lib_ens.ensemble(6, X_train, y_train, X_test, bags=i)
    metrics_ens = knn_lib_ens.metrics_ens(lib_pred, y_test)

    accuracies.append(metrics_ens[0])
    precisions.append(round(metrics_ens[2][0],3))
    recalls.append(round(metrics_ens[2][1], 3))
    errors.append(round(1-metrics_ens[0], 3))
    fs.append(round(metrics_ens[2][2], 3))

    conf_mat = metrics_ens[1]
    specif = conf_mat[1, 1]/(conf_mat[1, 1] + conf_mat[0, 1])

    specificity.append(round(specif, 3))

lib_metrics_obtained = [accuracies, precisions, recalls, errors, fs, specificity]
write_to_csv(lib_metrics_obtained, ["Accuracy", "Precision", "Recall", "Error", "F-score", "Specificity"], "Breast-Cancer-Lib-K=6")















#Change n_estimators
accuracies_ens = []
precisions_ens = []
recalls_ens = []
errors_ens = []
fs_ens = []

for i in range(2, 21):
    #from the accuracies obtained above, the highest accuracy is at K=8 or K=9
    knn_ensemble = KNNEnsemble("bagging", 8)
    knn_ensemble.fit(X_train, y_train)
    voted_pred = knn_ensemble.bagging(X_test, -1, bags=i)
    conf_matrix2 = metrics.confusion_matrix(y_test, voted_pred)
    accuracy2 = metrics.accuracy(conf_matrix2)
    error2 = metrics.error(conf_matrix2)
    precision2 = metrics.precision(conf_matrix2)
    recall2 = metrics.recall(conf_matrix2)
    fs = metrics.f_score()

    accuracies_ens.append(accuracy2)
    precisions_ens.append(precision2)
    recalls_ens.append(recall2)
    errors_ens.append(error2)
    fs_ens.append(fs)

metrics_obtained = [accuracies_ens, precisions_ens, recalls_ens, errors_ens, fs_ens]
write_to_csv(metrics_obtained, ["Accuracy", "Precision", "Recall", "Error", "F-score"], "Ens-From-Scratch at K=8")


# lib ens classifier
accuracies = []
precisions = []
recalls = []
errors = []
fs = []

for i in range(2, 21):
    knn_lib_ens = ScikitLearn()
    #from the accuracies obtained above, the highest accuracy is at K=11 or K=12
    lib_pred = knn_lib_ens.ensemble(11, X_train, y_train, X_test, bags=i)
    metrics_ens = knn_lib_ens.metrics_ens(lib_pred, y_test)

    accuracies.append(metrics_ens[0])
    precisions.append(round(metrics_ens[2][0],3))
    recalls.append(round(metrics_ens[2][1], 3))
    errors.append(round(1-metrics_ens[0], 3))
    fs.append(round(metrics_ens[2][2], 3))


lib_metrics_obtained = [accuracies, precisions, recalls, errors, fs]
write_to_csv(lib_metrics_obtained, ["Accuracy", "Precision", "Recall", "Error", "F-score"], "Ens-From-Lib at K=11")
