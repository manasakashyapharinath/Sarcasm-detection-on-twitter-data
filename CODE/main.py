# -*- coding: utf-8 -*-
"""
@Authors: Manasa Kashyap Harinath
	  Sravanthi Avasarala
          Siddhi Khandge

1)This file has three classifiers- Linear SVM and Logistic Regression and Random Forrest
2)The accuracy of each of these classifiers are calculated
3)Other metrics like F score, sensitivity and sensibility are also calculated using confusion matrix
"""
from nltk.classify import SklearnClassifier
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
import project
import preprocess
import numpy
import nltk
import random
import pandas as pd
from sklearn import metrics


def writeIntoMatrix(matrix):
    file = open('matrix.csv','w');
    line = '';
    for x in matrix:
        line += str(x) + ",";
    file.write(line);

def classifyUsingSVM(feature):
    # Define folds for cross validation
    kf = cross_validation.KFold(len(feature), n_folds=5, shuffle=False);
    features = numpy.array(feature);
    max2 = 0;
    SVMmodel = SklearnClassifier(SVC());
    #logreg = linear_model.LogisticRegression(C=1e5)
    
    for x,y in kf:
        train_set_fold = features[x];
        test_set_fold = features[y];
        train_set = list(train_set_fold);
        test_set = list(test_set_fold);
        # SV Classifier
        classifier2 = SklearnClassifier(SVC()).train(train_set);
	#classifier3=MultinomialNB()
        #classifier3.train(train_set)
        accuracy2 = nltk.classify.accuracy(classifier2,test_set)*100;
	#accuracy3 = nltk.classify.accuracy(classifier3,test_set)*100;
        # Use the best model       
        if accuracy2 > max2:
            SVMmodel = classifier2;
    return SVMmodel;

def ClassifyUsingLogisticRegression(feature):
     
     featuresetsUpdatedArray = numpy.array(feature);
     featureX=[]
     featureY=[]
     m,n=featuresetsUpdatedArray.shape
     for i in range(m):
	
		featureX.append(featuresetsUpdatedArray[i][0])
		featureY.append(int(featuresetsUpdatedArray[i][1]))

     #print(featureX)
     #print(featureY)
     train = pd.DataFrame(featureX)
     test=pd.DataFrame(featureY)
     #train=numpy.array(featureX)
     #test=numpy.array(featureY)
     X_train,X_test,Y_train,Y_test = train_test_split(train,test, test_size= 0.2, random_state = 42)
     #print(X_train)
     #print(Y_train)
     clf = LogisticRegression()
     clf.fit(X_train,Y_train)
     Y_predicted=clf.predict(X_test)
     ConfusionMatrix = confusion_matrix(Y_test,Y_predicted)
     sensitivity, specificity=calculateSensitivitySpecificity(ConfusionMatrix)
     #Y_predicted = model.predict(X_test)
     print('Classification Accuracy, sensitivity and specificity of Logistic Regression classifier is ');
     Accuracy = metrics.accuracy_score(Y_test,Y_predicted)*100
     f_score=f1_score(Y_test, Y_predicted, average='binary')
     #print('Classification Precision, Recall, F score and Support')
     #precision, recall, f_score, support = precision_recall_fscore_support(Y_test, Y_predicted, average='micro')
     print(Accuracy)
     print(sensitivity, specificity, f_score)
     #print(precision, recall, f_score, support)
		

def calculateSensitivitySpecificity(ConfusionMatrix):
    sensitivity = []
    specificity = []
    fscore=[]
    for i in range(ConfusionMatrix.shape[0]):
    	TP = float(ConfusionMatrix[i,i])  
    	FP = float(ConfusionMatrix[:,i].sum()) - TP  
    	FN = float(ConfusionMatrix[i,:].sum()) - TP  
    	TN = float(ConfusionMatrix.sum().sum()) - TP - FP - FN
    
	#print (TP,FP,FN,TN)
    	sensitivity.append(TP / (TP + FN))  #recall
    	specificity.append(TN / (TN + FP))  #Precison
	
    return sensitivity, sensitivity


def ClassifyUsingRandomForrest(feature):
     
     featuresetsUpdatedArray = numpy.array(feature);
     featureX=[]
     featureY=[]
     m,n=featuresetsUpdatedArray.shape
     for i in range(m):
	
		featureX.append(featuresetsUpdatedArray[i][0])
		featureY.append(int(featuresetsUpdatedArray[i][1]))

    # print(featureX)
    # print(featureY)
     train = pd.DataFrame(featureX)
     test=pd.DataFrame(featureY)
     #train=numpy.array(featureX)
     #test=numpy.array(featureY)
     X_train,X_test,Y_train,Y_test = train_test_split(train,test, test_size= 0.2, random_state = 42)
     #print(X_train)
     #print(Y_train)
     clf = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
     clf.fit(X_train,Y_train)
     Y_predicted=clf.predict(X_test)
     ConfusionMatrix = confusion_matrix(Y_test,Y_predicted)
     sensitivity, specificity=calculateSensitivitySpecificity(ConfusionMatrix)
     #Y_predicted = model.predict(X_test)
     print('Classification Accuracy, sensitivity,specificity and f score of Random Forrest classifier is ');
     Accuracy = metrics.accuracy_score(Y_test,Y_predicted)*100
     f_score=f1_score(Y_test, Y_predicted, average='binary') 
     #print('Classification Precision, Recall, F score and Support')
     #precision, recall, f_score, support = precision_recall_fscore_support(Y_test, Y_predicted, average='micro')
     print(Accuracy)
     print(sensitivity, specificity, f_score)
     

def CalculateAccuracyOfSVM(feature):
    featuresetsUpdatedArray = numpy.array(feature);
    featureX=[]
    featureY=[]
    m,n=featuresetsUpdatedArray.shape
    for i in range(m):
	
		featureX.append(featuresetsUpdatedArray[i][0])
		featureY.append(int(featuresetsUpdatedArray[i][1]))

    # print(featureX)
    # print(featureY)
    train = pd.DataFrame(featureX)
    test=pd.DataFrame(featureY)
     #train=numpy.array(featureX)
     #test=numpy.array(featureY)
    X_train,X_test,Y_train,Y_test = train_test_split(train,test, test_size= 0.2, random_state = 42)
     #print(X_train)
     #print(Y_train)
     #clf = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
    clf=LinearSVC()
    clf.fit(X_train,Y_train)
    Y_predicted=clf.predict(X_test)
    ConfusionMatrix = confusion_matrix(Y_test,Y_predicted)
     #Y_predicted = model.predict(X_test)
    sensitivity, specificity=calculateSensitivitySpecificity(ConfusionMatrix)
     #Y_predicted = model.predict(X_test)
    print('Classification Accuracy, sensitivity , specificity and f score of SVM classifier is ');
    Accuracy = metrics.accuracy_score(Y_test,Y_predicted)*100
    f_score=f1_score(Y_test, Y_predicted, average='binary') 
    #print('Classification Precision and Recall')
    #precision, recall, f_score, support = precision_recall_fscore_support(Y_test, Y_predicted, average='micro')
    print(Accuracy)
    print(sensitivity,specificity,f_score )
    



def main():
    # Reading the training file
    file = open('Mature.train','r');
    count = 0;
    lines = [];
    while count < 990:
        try:
            line = (file.readline());
            if line is not '':
                lines.append(line);
                line = '';
        except UnicodeDecodeError:
            continue;
        count += 1;
    trainData = [];
    targetWord = (lines[0].split('\t'))[0];
    print('\nTarget Word is ',targetWord);
    # Preprocessing the data    
    for line in lines:
        x = line.split("\t");
        try:
            trainData.append((preprocess.preprocess(line),x[1]));
        except IndexError:
            print();
    
    m = int(len(trainData)*0.8);
    random.seed(2);
    random.shuffle(trainData);
    testData = trainData[m:];
    data = trainData[:m];
    trainData = data;
   
    print('\nLoading feature vectors pre-trained on Training data');
    n = len(trainData);
    print(trainData)
    print(n);
    print('\nComputing Kernel Matirx');
    featureList = [];
    kMatrix = [[0 for x in range(n)] for x in range(n)];
    for i in range(n):
        print(i);
        iDictionary = {};
        for j in range(n):
            kMatrix[i][j] = project.mvme(trainData[i],trainData[j]);
            iDictionary[j] = kMatrix[i][j];
        featureList.append([iDictionary,(trainData[i])[1]]);
    # Write the kernel matrix to CSV file
    writeIntoMatrix(kMatrix);
    
    # Train the classification model
    print('\nTraining the SVM, LR and RF models using the kernel matrix');
    classifier = classifyUsingSVM(featureList);   # SVM Classifier
    ClassifyUsingLogisticRegression(featureList)  # Logistic Regression Classifier
    ClassifyUsingRandomForrest(featureList)  #Random Forrest Classifier
    CalculateAccuracyOfSVM(featureList) #Calculating accuracy of SVM classifier
    print('\n\n');
    n  = len(testData);
    featureList = [];
    kMatrix = [[0 for x in range(n)] for x in range(n)];
    for i in range(n):
        print(i);
        iDictionary = {};
        for j in range(n):
            kMatrix[i][j] = project.mvme(testData[i],testData[j]);
            iDictionary[j] = kMatrix[i][j];
        featureList.append([iDictionary,(testData[i])[1]]);
    # Test the classification model
    #print('Classification Accuracy of SVM classifier is ');
    #print(nltk.classify.accuracy(classifier,featureList)*100);
        
main()
