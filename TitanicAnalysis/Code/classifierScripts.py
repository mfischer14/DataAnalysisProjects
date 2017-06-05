def getPredictions(trainedClassifier, features_test):
    return trainedClassifier.predict(features_test)

def classifyTree(features_learn, features_test):
    from sklearn.tree import DecisionTreeClassifier

    clf = DecisionTreeClassifier()
    return clf.fit(features_learn, features_test)


def classifyNaiveBayes(features_learn, features_test):
    from sklearn.naive_bayes import GaussianNB

    clf = GaussianNB()
    return clf.fit(features_learn, features_test)

def classifySVM(features_learn, features_test):
    from sklearn.svm import SVC

    clf = SVC(C = 10000, gamma = 10)
    return clf.fit(features_learn, features_test)

def classifierAccuracy(predicted_labels, labels_test):
    from sklearn.metrics import accuracy_score
    return accuracy_score(predicted_labels, labels_test)
