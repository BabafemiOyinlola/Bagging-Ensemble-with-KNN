import numpy as np
from preprocess import *
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import *
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, LabelBinarizer


class ScikitLearn:
    def KNNClassifier(self, k, X_train, y_train, X_test):      
        y_train = LabelEncoder().fit_transform(y_train)
        X_train = np.array(X_train)

        knn = KNeighborsClassifier(k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        return y_pred

    def ensemble(self, k, X_train, y_train, X_test, bags):
        bag_clf = BaggingClassifier(KNeighborsClassifier(n_neighbors=k), bootstrap=True, n_estimators=bags)
        bag_clf.fit(X_train, y_train)
        y_pred = bag_clf.predict(X_test)
        return y_pred

    def metrics_base(self, y_pred, y_test):
        y_test = LabelEncoder().fit_transform(y_test)
        accuracy = round(accuracy_score(y_test, y_pred), 2)
        conf_matrix = confusion_matrix(y_test, y_pred)
        pre, rec, fscor = precision_recall_fscore_support(y_test, y_pred)
        return(accuracy, conf_matrix, pre, rec, fscor)

    def metrics_ens(self, y_pred, y_test):
        y_test = LabelEncoder().fit_transform(y_test)
        y_pred = LabelEncoder().fit_transform(y_pred)
        accuracy = round(accuracy_score(y_test, y_pred), 2)
        conf_matrix = confusion_matrix(y_test, y_pred) 
        pre= precision_recall_fscore_support(y_test, y_pred, average="weighted")
        return(accuracy, conf_matrix, pre)

    def k_foldcross_validation(self, data,label_pos, K, n_est, n=10):
        k_fold = KFold(n_splits=n, shuffle=True, random_state=0)
        bag_clf = BaggingClassifier(KNeighborsClassifier(n_neighbors=K), bootstrap=True, n_estimators=n_est)      

        features, labels = [], []
        if label_pos == -1:
            for i in range(len(data)):     
                labels.append(data[i][-1])
                features.append(data[i][0:(len(data[i])-1)])

        elif label_pos == 0:
            for i in range(len(data)):
                labels.append(data[i][0])
                features.append(data[i][1:len(data[i])])

        preprocess = Preprocess()
        features = preprocess.normalise_data(features)

        accuracy =  np.around(cross_val_score(bag_clf, features, labels, scoring='accuracy', cv = k_fold), 3)
        
        labels = LabelBinarizer().fit_transform(labels)

        recall = np.around(cross_val_score(bag_clf, features, labels, scoring='recall', cv = k_fold), 3)
        precision = np.around(cross_val_score(bag_clf, features, labels, scoring='precision', cv = k_fold), 3)
        fs = np.around(cross_val_score(bag_clf, features, labels, scoring='f1_weighted', cv = k_fold), 3)
        scorer = make_scorer(self.specificity)    
        specifici = np.around(cross_val_score(bag_clf, features, labels, scoring=scorer, cv = k_fold), 3) 
        error = np.round(1 - accuracy, 3)

        metrics = [accuracy, precision, recall, error, fs, specifici]
        return metrics

    def specificity(self, y_test, y_pred):
        y_test = np.reshape(y_test, np.shape(y_pred))
        conf_matrix = confusion_matrix(y_test, y_pred)
        tn = conf_matrix[1, 1]
        fp = conf_matrix[0, 1]
        speci = tn/(tn+fp)
        return speci