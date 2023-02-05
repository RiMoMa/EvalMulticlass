from keras import layers
from keras import models
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from PIL import Image as ImageOP
import matplotlib.pyplot as plt
import glob
from keras.preprocessing import image
from itertools import product
from pathlib import Path
import os.path
import h5py
from sklearn.model_selection import train_test_split
import os
#os.environ["CUDA_VISIBLE_DEVICES"]=""
#import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
sys.path.append('/home/ricardo/PycharmProjects/pykernels-master/pykernels/')
sys.path.append('/home/ricardo/PycharmProjects/pykernels-master/pykernels/graph/')


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
import pykernels
from pykernels.regular import Chi2,Cossim,GeneralizedHistogramIntersection
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.cluster import KMeans
h = .02  # step size in the mesh
import pickle
#pathTrainArray = 'XtrainFeatures2019.h5'
#pathTestArray = 'XtestFeatures2019.h5'

#h5f = h5py.File(pathTrainArray, 'r')
#Xtrain = h5f['XtrainAuto'][:]
#Ytrain = h5f['Y_train_X20'][:]
#h5f.close()
#print('Vector TRAINING LOAD!')

"""PARTE DE TEST"""

names = ["Nearest Neighbors",
         "Linear SVM",
         "RBF SVM",
         "Chi2 SVM",
         "Cosine SVM",
         "GHK SVM",
         "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA", "LogisticRegression", "GradientBoosting",
         #"Voting1vs3",
       #  "Voting1v2", "NewVoting",
       #  "onevsallVoting"
         ]

classifiers = [
    KNeighborsClassifier(3),
    linear_model.SGDClassifier(max_iter=1000, tol=1e-3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    SVC(kernel=Chi2(), C=1),
    SVC(kernel=Cossim(), C=1),
    SVC(kernel=GeneralizedHistogramIntersection(), C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=10),
    RandomForestClassifier(max_depth=1, n_estimators=12, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(priors=None, var_smoothing=1e-04),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression(solver='lbfgs'),
    GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
]




#files = glob.glob('/home/ricardo/Documentos/experimentosMay/')
files=sorted(glob.glob('/home/ricardo/Downloads/enviarRMM.mat'), key=os.path.basename)
import scipy.io as sio
from sklearn.model_selection import cross_val_score
Kfold = 10

for fileTrain in files:

    print(fileTrain)
    fileTR = fileTrain
    fileTE = fileTrain[0:10] + 'Test_' + fileTrain[12::]
    print(fileTE)

    file_mat=sio.loadmat(fileTR)
    Xtrain = file_mat['FeaturesCasoSurvival_FNanF']
    Ytrain = file_mat['YF']
    topFeatIdx = file_mat['topFeatIdx']
    topFeatIdx = topFeatIdx[:]
    Xtrain = Xtrain[:,topFeatIdx[0]]
    Xtrain = Xtrain.astype(np.float32)
    #search and eliminate Nan rows
    RowsNan = []
    for n in range(0, Xtrain.shape[0]):
        if np.isinf(Xtrain[n, :]).any() == True:
            RowsNan.append(n)
            print('value:', n, np.isinf(Xtrain[n, :]).any() == True)
    Xtrain = np.delete(Xtrain,RowsNan,0)
    Ytrain = np.delete(Ytrain,RowsNan,0)

    #Ytrain = Ytrain[:, :]
    #Xtrain = Xtrain
   # Ytrain = np.reshape(Ytrain,len(Ytrain))

    my_file = Path(fileTE)
    if my_file.is_file():
        file_mat=sio.loadmat(fileTE)
        XtestF = file_mat['all_features']
        Ytest = file_mat['all_labels']
        # search and eliminate Nan rows
        RowsNan = []
        for n in range(0, XtestF.shape[0]):
            if np.isinf(XtestF[n, :]).any() == True:
                RowsNan.append(n)
                print('value:', n, np.isinf(XtestF[n, :]).any() == True)
        XtestF = np.delete(XtestF, RowsNan, 0)
        Ytest = np.delete(Ytest, RowsNan, 0)

    else:
        XtestF = Xtrain
        Ytest = Ytrain


    Ytest = np.reshape(Ytest,len(Ytest))

    #Xtrain = Xtrain[:,1:]


    Y_train_X20 = Ytrain[:,0]
    Y_test_X20 = Ytest

    """Find arrays"""
    index = Y_train_X20 == 1
    Xtrain1=Xtrain[index,:]
    Ytrain1 = Y_train_X20[index]

    index = Y_train_X20 == 0
    Xtrain2=Xtrain[index,:]
    Ytrain2 = Y_train_X20[index]


    index = Y_test_X20 == 1
    Xtest1=XtestF[index,:]
    Ytest1 = Y_test_X20[index]
    index = Y_test_X20 == 0
    Xtest2=XtestF[index,:]
    Ytest2 = Y_test_X20[index]


    Name_test = ["1Vs2","2Vs1"]

    for z in range(len(Name_test)):
        print(z)
        if z==0:
            Xtotal = np.concatenate((Xtrain1,Xtrain2), axis=0)
            Ytotal = np.concatenate((Ytrain1, Ytrain2 * 0), axis=0)
            Xtesttotal = np.concatenate((Xtest1, Xtest2), axis=0)
            Ytesttotal = np.concatenate((Ytest1, Ytest2 * 0), axis=0)

        elif z == 1:
            Xtotal = np.concatenate((Xtrain1, Xtrain2), axis=0)
            Ytotal = np.concatenate((Ytrain1*0 , Ytrain2+1), axis=0)
            Xtesttotal = np.concatenate((Xtest1, Xtest2), axis=0)
            Ytesttotal = np.concatenate((Ytest1*0 , Ytest2+1 ), axis=0)



        X_train = Xtotal
        X_test = Xtesttotal
        y_train = Ytotal
        y_test = Ytesttotal
        X_train = preprocessing.MinMaxScaler().fit_transform(Xtotal)
        X_test = preprocessing.MinMaxScaler().fit_transform(Xtesttotal)

    #    X_train = preprocessing.scale(X_train)
  #      X_test = preprocessing.scale(X_test)
        #X_train = preprocessing.MinMaxScaler().fit_transform(Xtotal)
        #X_test = preprocessing.MinMaxScaler().fit_transform(Xtesttotal)
        y_train = Ytotal
        y_test = Ytesttotal




        # iterate over classifiers
        score_KF=[]
        for name, clf in zip(names, classifiers):
          #  ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
            print(clf)
            #clf.fit(X_train, y_train)
            #score = clf.score(X_test, y_test)
            scores = cross_val_score(clf, X_train, y_train, cv=5,scoring='f1_macro',error_score=np.nan)
            score = scores.mean()
            print(name,': ',score)
            #y_pred = clf.predict(X_test)
            #confusion = confusion_matrix(y_test,y_pred)
            #print(confusion)
            #fscore_aux = f1_score(y_test, y_pred, average='macro')
            score_KF.append(score)
            print('score: ', score)
        indice = score_KF.index(max(score_KF))
        print(Name_test[z],'Best Classifier:\n',names[indice],'\nScore KFOLD: ', score_KF[indice])
        print('FINAL RESULT: ',Name_test[z])
        clf = classifiers[indice].fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print(names[indice],'Test SET: ', score)
        y_pred = clf.predict(X_test)
        confusion = confusion_matrix(y_test, y_pred)
        print(confusion)
        print('F-score Test:',f1_score(y_test, y_pred, average='macro'))











































