''' Main script to analyze the data '''

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # package to embellish graphs / plots
import matplotlib.pyplot as plt
import pylab as pl
from statisticalDescription import describeData
from datasetCleanup import *
from classifierScripts import *

# Import the train.csv file into a dataframe
titanic_train = pd.read_csv('../input/train.csv')
titanic_test = pd.read_csv('../input/test.csv')

# Describe the data first
describeData(titanic_train)
#performCleanup()
titanic_train = performCleanup(titanic_train)
titanic_test = performCleanup(titanic_test)

def writeToFile(pred, featTest, fileName):
    results = pd.DataFrame(columns=['PassengerId', 'Survived'])
    results['PassengerId'] = featTest['PassengerId']
    results['Survived'] = pred
    #numpy.savetxt("../output/TreePredictions01.csv", pred, delimiter=",")
    results.to_csv("../output/{}".format(fileName), header=True, index=False)

# First Prediction: Based on Gender
def evaluateMethodologies(featTrain, labelTrain):
    features_train, labels_train, features_test, labels_test = generateTestSubSample(featTrain, labelTrain)

    ### Presentation of the split
    # print "ENTIRE DATA FEATURES"
    # describeData(titanicFeatures_train)
    # print "ENTIRE DATA LABELS"
    # describeData(titanicLabels_train)
    #
    # print "TRAIN DATA FEATURES"
    # describeData(features_train)
    # print "TRAIN DATA LABELS"
    # describeData(labels_train)
    #
    # print "TEST DATA FEATURES"
    # describeData(features_test)
    # print "TEST DATA LABELS"
    # describeData(labels_test)

    clf = classifyNaiveBayes(features_train, labels_train)
    pred = clf.predict(features_test)
    print "Naive Bayes Accuracy: {}:".format(classifierAccuracy(pred, list(labels_test)))

    clf = classifySVM(features_train, labels_train)
    pred = clf.predict(features_test)
    print "SVM Accuracy: {}:".format(classifierAccuracy(pred, list(labels_test)))

    clf = classifyTree(features_train, labels_train)
    pred = clf.predict(features_test)
    print "Tree Accuracy: {}:".format(classifierAccuracy(pred, list(labels_test)))

def predictionMethod_ByGender(featTrain, labelTrain, featTest):
    ### Score: 0.76555
    pred = (featTest['Sex'] == 0) * 1
    writeToFile(pred, featTest, "GenderPrediction01.csv")

def predictionMethod_UsingTree(featTrain, labelTrain, featTest):
    ### Score: 0.69856
    clf = classifyTree(featTrain, labelTrain)
    pred = clf.predict(featTest)
    writeToFile(pred, featTest, "TreePrediction01.csv")

def predictionMethod_UsingNaiveBayes(featTrain, labelTrain, featTest):
    ### Score: 0.73206
    clf = classifyNaiveBayes(featTrain, labelTrain)
    pred = clf.predict(featTest)
    writeToFile(pred, featTest, "NaiveBayesPrediction01.csv")

def predictionMethod_UsingSVM(featTrain, labelTrain, featTest):
    ### Score: 0.73206
    clf = classifySVM(featTrain, labelTrain)
    pred = clf.predict(featTest)
    writeToFile(pred, featTest, "SVMPrediction01.csv")

titanicFeatures_train, titanicLabels_train = splitFeaturesFromLabels(titanic_train)
#evaluateMethodologies(titanicFeatures_train, titanicLabels_train)
predictionMethod_ByGender(titanicFeatures_train, titanicLabels_train, titanic_test)
predictionMethod_UsingTree(titanicFeatures_train, titanicLabels_train, titanic_test)
predictionMethod_UsingNaiveBayes(titanicFeatures_train, titanicLabels_train, titanic_test)
predictionMethod_UsingSVM(titanicFeatures_train, titanicLabels_train, titanic_test)
