from KNN import *

def read_csv(filepath, header=True):
    dataset = []
    read = open(filepath, "r")
    content = read.readlines()

    if(len(content) == 0):
            return
    else:
        if header == True:
            for line in range(1, len(content)):
                dataset.append(content[line].rstrip().split(','))
        elif header == False:
            for line in range(len(content)):
                row = []
                dataset.append(content[line].rstrip().split(','))
        read.close()
        return dataset

#Split into train and test set
def split_data(data, label_pos, ratio=0.3):
    train, test = [], []

    test_num = int(ratio * len(data))
    for i in range(test_num):
        index = random.randint(0, (len(data)-1))
        item = data[index]
        test.append(item)
        del data[index]

    for i in range(len(data)):
        index = random.randint(0, (len(data)-1))
        item = data[index]
        train.append(item)
        del data[index]
    
    X_train, y_train, X_test, y_test = [], [], [], []

    if label_pos == -1:
        for i in range(len(train)):
            feat = train[i]
            feat = feat[0:len(train[0])-1]
            feat = [float(i) for i in feat]
            X_train.append(feat)
            y_train.append(train[i][-1])

        for i in range(len(test)):
            feat = test[i]
            feat = feat[0:len(test[0])-1]
            feat = [float(i) for i in feat]
            X_test.append(feat)
            y_test.append(test[i][-1])


    elif label_pos == 0:
        # X_train = train[1:len(train[0])]   
        # y_train = train[:0]
        # X_test = test[1:len(test[0])]   
        # y_test = test[:0]

        for i in range(len(train)):
            feat = train[i]
            feat = feat[1:len(train[0])]
            feat = [float(i) for i in feat]
            X_train.append(feat)
            y_train.append(train[i][0])

        for i in range(len(test)):
            feat = test[i]
            feat = feat[1:len(test[0])]
            feat = [float(i) for i in feat]
            X_test.append(feat)
            y_test.append(test[i][0])

    return(X_train, y_train, X_test, y_test)


data = read_csv("/Users/oyinlola/Desktop/MSc Data Science/SCC403 - Data Mining/iris.csv", True)
X_train, y_train, X_test, y_test = split_data(data, -1)

# knn = KNN(8)
# knn.fit(X_train, y_train)
# y_pred = knn.predict(X_test)
# conf_matrix = knn.confussion_matrix(y_test, y_pred)

knn_ensemble = KNNEnsemble("bagging", 4)
knn_ensemble.fit(X_train, y_train)
voted = knn_ensemble.bagging(X_test, -1)

print("Done")