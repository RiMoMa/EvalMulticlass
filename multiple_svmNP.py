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

import pickle
pathTrainArray = 'XtrainFeatures2019.h5'
pathTestArray = 'XtestFeatures2019.h5'

h5f = h5py.File(pathTrainArray, 'r')
Xtrain = h5f['XtrainAuto'][:]
Ytrain = h5f['Y_train_X20'][:]
h5f.close()
print('Vector TRAINING LOAD!')

"""PARTE DE TEST"""

h5f = h5py.File(pathTestArray, 'r')
XtestF = h5f['XtestAuto'][:]
Ytest = h5f['Y_test'][:]
h5f.close()
print('Vector TEST LOAD!')

print('Feature Extraction from Test READY!')
"""MATLAB"""

import scipy.io as sio

file_mat=sio.loadmat('/home/ricardo/Escritorio/TrainNM.mat')
Xtrain = file_mat['Xtrain']
Ytrain = file_mat['Ytrain']

file_mat=sio.loadmat('/home/ricardo/Escritorio/TestNM.mat')

XtestF = file_mat['Xtest']
Ytest = file_mat['Ytest']

Ytrain = np.reshape(Ytrain,len(Ytrain))
Ytest = np.reshape(Ytest,len(Ytest))

Xtrain = Xtrain[:,:]
XtestF = XtestF[:,:]

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
from sklearn.cluster import KMeans
h = .02  # step size in the mesh


Y_train_X20 = Ytrain
Y_test_X20 = Ytest

"""Find arrays"""
index = Y_train_X20 == 1
Xtrain1=Xtrain[index,:]
Ytrain1 = Y_train_X20[index]
index = Y_train_X20 == 1
Xtrain2=Xtrain[index,:]
Ytrain2 = Y_train_X20[index]

index = Y_train_X20 == 1
Xtrain2=Xtrain[index]
Ytrain2 = Y_train_X20[index]

clus = 80
kmeans = KMeans(n_clusters=clus, random_state=0,max_iter=1000).fit(Xtrain2)
Ytrain2K=Ytrain2[0:clus]
kmeans =kmeans.cluster_centers_

index = Y_train_X20 == 0
Xtrain3=Xtrain[index,:]
Ytrain3 = Y_train_X20[index]


index = Y_test_X20 == 1
Xtest1=XtestF[index,:]
Ytest1 = Y_test_X20[index]
index = Y_test_X20 == 1
Xtest2=XtestF[index,:]
Ytest2 = Y_test_X20[index]
index = Y_test_X20 == 0
Xtest3=XtestF[index,:]
Ytest3 = Y_test_X20[index]



Name_test = ["3VS.1&2","1VS.2&3","1Vs2","1Vs3","2Vs3","2VS.1&3"]


Xtotal = np.concatenate((Xtrain1, Xtrain3), axis=0)
Ytotal = np.concatenate((Ytrain1, Ytrain3), axis=0)
Xtesttotal = np.concatenate((Xtest1, Xtest3), axis=0)
Ytesttotal = np.concatenate((Ytest1, Ytest3), axis=0)




X_train = Xtotal
X_test = Xtesttotal
y_train = Ytotal
y_test = Ytesttotal
X_train = preprocessing.MinMaxScaler().fit_transform(Xtotal)
X_test = preprocessing.MinMaxScaler().fit_transform(Xtesttotal)
y_train = Ytotal
y_test = Ytesttotal




names = ["Nearest Neighbors", "Linear SVM","RBF","svm cosine",
         "histogram intersection", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA","LogisticRegression","GradientBoosting","Voting1vs3","Voting1v2","NewVoting","onevsallVoting"]

classifiers = [
    KNeighborsClassifier(3),
    #SVC(kernel="linear", C=0.025),
    SVC(kernel="linear", C=0.1),
    SVC(gamma=2, C=1),
   # SVC(kernel=Chi2(), C=1),
    SVC(kernel=Cossim(), C=1),
    SVC(kernel=GeneralizedHistogramIntersection(), C=1),
    #GaussianProcessClassifier(1.0 * RBF(1.0)),
    #DecisionTreeClassifier(max_depth=2),
    MLPClassifier(alpha=1),
#    RandomForestClassifier(max_depth=1, n_estimators=2, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(priors=None, var_smoothing=1e-04),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression(solver='lbfgs'),
    GradientBoostingClassifier(n_estimators=10, learning_rate=1.0,max_depth=1, random_state=0),
   # VotingClassifier(estimators=[('QDA', QuadraticDiscriminantAnalysis()), ('NB', GaussianNB()),('Dtree', DecisionTreeClassifier(max_depth=5))], voting='hard')
    VotingClassifier(estimators=[('LSVC',SVC(kernel="linear", C=0.025)),
                                 ('GraBos', AdaBoostClassifier()),
                                 ('NB', RandomForestClassifier(max_depth=5, n_estimators=30, max_features=2))], voting='hard'),

    VotingClassifier(estimators=[('QDA', MLPClassifier(alpha=1)),
                                 ('GraBos', GradientBoostingClassifier(n_estimators=30, learning_rate=1.0, max_depth=1, random_state=0)),
                                 ('Lr',LogisticRegression(solver='lbfgs')),
                                 ('Dtree', DecisionTreeClassifier(max_depth=10))], voting='hard'),

    VotingClassifier(estimators=[('QDA', QuadraticDiscriminantAnalysis()),
                                 ('GraBos', AdaBoostClassifier()),
                                 ('Lr', MLPClassifier(alpha=1)),
                                 ('Dtree', DecisionTreeClassifier(max_depth=10))], voting='hard'),

    OneVsOneClassifier(GaussianNB())
]

# iterate over classifiers
fscore=[]
for name, clf in zip(names, classifiers):
  #  ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(name,': ',score)
    y_pred = clf.predict(X_test)
    confusion = confusion_matrix(y_test,y_pred)
    print(confusion)
    fscore_aux = f1_score(y_test, y_pred, average='macro')
    fscore.append(fscore_aux)
    print('fscore: ', fscore_aux)
indice = fscore.index(max(fscore))
print(Name_test[z],'Best Classifier:\n',names[indice],'\nFscore: ', fscore[indice])
print('FINAL RESULT: ',Name_test[z])
clf = classifiers[indice].fit(X_train, y_train)
score = clf.score(X_test, y_test)
print(names[indice],': ', score)
y_pred = clf.predict(X_test)
confusion = confusion_matrix(y_test, y_pred)
print(confusion)










































