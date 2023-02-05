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


import numpy as np
import h5py
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import scipy.io as sio

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

import scipy.io as sio
file_mat=sio.loadmat('/home/ricardo/Documentos/SYNC/ExperimentosGrafoHaralickS/trainNorm_augmented_Words50block_24L3_END.mat')
Xtrain = file_mat['X']
Ytrain = file_mat['Y']
file_mat=sio.loadmat('/home/ricardo/Documentos/SYNC/ExperimentosGrafoHaralickS/testNorm_augmented_Words50block_24L3_END.mat')
XtestF = file_mat['Xtest']
Ytest = file_mat['Ytest']
Ytrain = np.reshape(Ytrain,len(Ytrain))
Ytest = np.reshape(Ytest,len(Ytest))
Xtrain = Xtrain[:,:]
XtestF = XtestF[:,:]


Y_train_X20 = Ytrain
Y_test_X20 = Ytest

"""Find arrays"""
index = Y_train_X20 == 1
Xtrain1=Xtrain[index,:]
Ytrain1 = Y_train_X20[index]
index = Y_train_X20 == 2
Xtrain2=Xtrain[index,:]
Ytrain2 = Y_train_X20[index]

index = Y_train_X20 == 2
Xtrain2=Xtrain[index]
Ytrain2 = Y_train_X20[index]

clus = 80
kmeans = KMeans(n_clusters=clus, random_state=0,max_iter=1000).fit(Xtrain2)
Ytrain2K=Ytrain2[0:clus]
kmeans =kmeans.cluster_centers_

index = Y_train_X20 == 3
Xtrain3=Xtrain[index,:]
Ytrain3 = Y_train_X20[index]


index = Y_test_X20 == 1
Xtest1=XtestF[index,:]
Ytest1 = Y_test_X20[index]
index = Y_test_X20 == 2
Xtest2=XtestF[index,:]
Ytest2 = Y_test_X20[index]
index = Y_test_X20 == 3
Xtest3=XtestF[index,:]
Ytest3 = Y_test_X20[index]

Xtotal = np.concatenate((Xtrain1, kmeans, Xtrain3), axis=0)
Ytotal = np.concatenate((Ytrain1 * 0, Ytrain2K * 0, Ytrain3), axis=0)
Xtesttotal = np.concatenate((Xtest1, Xtest2, Xtest3), axis=0)
Ytesttotal = np.concatenate((Ytest1 , Ytest2 , Ytest3), axis=0)

X_train = Xtotal
X_test = Xtesttotal
y_train = Ytotal
y_test = Ytesttotal
X_train = preprocessing.MinMaxScaler().fit_transform(Xtotal)
X_test = preprocessing.MinMaxScaler().fit_transform(Xtesttotal)
X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)

y_train = Ytotal
y_test = Ytesttotal

#clf = GaussianProcessClassifier(1.0 * RBF(0.7))
clf = VC(kernel=GeneralizedHistogramIntersection(), C=1)
clf.fit(X_train, y_train)

score = clf.score(X_test, y_test)
print('3Vs12.',': ',score)
y_pred = clf.predict(X_test)

confusion = confusion_matrix(y_test, y_pred)
print(confusion)
fscore_aux = f1_score(y_test, y_pred, average='macro')
print('fscore: ', fscore_aux)



y_test2= y_test[y_pred==0]


file_mat=sio.loadmat('/home/ricardo/Documentos/experimentosMay/Escheme/pasarLap/trainNorm_augmented_Words300block_24L2_END0.mat')
Xtrain = file_mat['X']
Ytrain = file_mat['Y']
file_mat=sio.loadmat('/home/ricardo/Documentos/experimentosMay/Escheme/pasarLap//testNorm_augmented_Words300block_24L2_END0.mat')
XtestF = file_mat['Xtest']
Ytest = file_mat['Ytest']
Ytrain = np.reshape(Ytrain,len(Ytrain))
Ytest = np.reshape(Ytest,len(Ytest))
Xtrain = Xtrain[:,:]
XtestF = XtestF[:,:]



Y_train_X20 = Ytrain
Y_test_X20 = Ytest

"""Find arrays"""
index = Y_train_X20 == 1
Xtrain1=Xtrain[index,:]
Ytrain1 = Y_train_X20[index]
index = Y_train_X20 == 2
Xtrain2=Xtrain[index,:]
Ytrain2 = Y_train_X20[index]

index = Y_train_X20 == 2
Xtrain2=Xtrain[index]
Ytrain2 = Y_train_X20[index]

clus = 80
kmeans = KMeans(n_clusters=clus, random_state=0,max_iter=1000).fit(Xtrain2)
Ytrain2K=Ytrain2[0:clus]
kmeans =kmeans.cluster_centers_

index = Y_train_X20 == 3
Xtrain3=Xtrain[index,:]
Ytrain3 = Y_train_X20[index]


index = Y_test_X20 == 1
Xtest1=XtestF[index,:]
Ytest1 = Y_test_X20[index]
index = Y_test_X20 == 2
Xtest2=XtestF[index,:]
Ytest2 = Y_test_X20[index]
index = Y_test_X20 == 3
Xtest3=XtestF[index,:]
Ytest3 = Y_test_X20[index]

Xtotal = np.concatenate((Xtrain1, kmeans, Xtrain3), axis=0)
Ytotal = np.concatenate((Ytrain1, Ytrain2K * 0, Ytrain3 * 0), axis=0)
Xtesttotal = np.concatenate((Xtest1, Xtest2, Xtest3), axis=0)
Ytesttotal = np.concatenate((Ytest1, Ytest2 , Ytest3 ), axis=0)



X_train = Xtotal
X_test = Xtesttotal
y_train = Ytotal
y_test = Ytesttotal
X_train = preprocessing.MinMaxScaler().fit_transform(Xtotal)
X_test = preprocessing.MinMaxScaler().fit_transform(Xtesttotal)
X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)

y_train = Ytotal
y_test2 = Ytesttotal


clf2 =  MLPClassifier(alpha=0.8)

clf2.fit(X_train, y_train)
X_test2 = X_test
y_pred2 =clf2.predict(X_test2)

confusion = confusion_matrix(y_test2, y_pred2)
print(confusion)
fscore_aux = f1_score(y_test, y_pred2, average='macro')
print('fscore: ', fscore_aux)


file_mat=sio.loadmat('/home/ricardo/Documentos/experimentosMay/Escheme/pasarLap/trainNorm_augmented_Words50block_96L0_END0.mat')
Xtrain = file_mat['X']
Ytrain = file_mat['Y']
file_mat=sio.loadmat('/home/ricardo/Documentos/experimentosMay/Escheme/pasarLap/testNorm_augmented_Words50block_96L0_END0.mat')
XtestF = file_mat['Xtest']
Ytest = file_mat['Ytest']
Ytrain = np.reshape(Ytrain,len(Ytrain))
Ytest = np.reshape(Ytest,len(Ytest))
Xtrain = Xtrain[:,:]
XtestF = XtestF[:,:]



Y_train_X20 = Ytrain
Y_test_X20 = Ytest

"""Find arrays"""
index = Y_train_X20 == 1
Xtrain1=Xtrain[index,:]
Ytrain1 = Y_train_X20[index]
index = Y_train_X20 == 2
Xtrain2=Xtrain[index,:]
Ytrain2 = Y_train_X20[index]

index = Y_train_X20 == 2
Xtrain2=Xtrain[index]
Ytrain2 = Y_train_X20[index]

clus = 80
kmeans = KMeans(n_clusters=clus, random_state=0,max_iter=1000).fit(Xtrain2)
Ytrain2K=Ytrain2[0:clus]
kmeans =kmeans.cluster_centers_

index = Y_train_X20 == 3
Xtrain3=Xtrain[index,:]
Ytrain3 = Y_train_X20[index]


index = Y_test_X20 == 1
Xtest1=XtestF[index,:]
Ytest1 = Y_test_X20[index]
index = Y_test_X20 == 2
Xtest2=XtestF[index,:]
Ytest2 = Y_test_X20[index]
index = Y_test_X20 == 3
Xtest3=XtestF[index,:]
Ytest3 = Y_test_X20[index]

Xtotal = np.concatenate((Xtrain1, kmeans, Xtrain3), axis=0)
Ytotal = np.concatenate((Ytrain1 * 0, Ytrain2K, Ytrain3 * 0), axis=0)
Xtesttotal = np.concatenate((Xtest1, Xtest2, Xtest3), axis=0)
Ytesttotal = np.concatenate((Ytest1 , Ytest2, Ytest3 ), axis=0)

X_train = Xtotal
X_test = Xtesttotal
y_train = Ytotal
y_test3 = Ytesttotal
X_train = preprocessing.MinMaxScaler().fit_transform(Xtotal)
X_test = preprocessing.MinMaxScaler().fit_transform(Xtesttotal)
X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)


clf3 = SVC(kernel=GeneralizedHistogramIntersection(), C=1)


clf3.fit(X_train, y_train)
y_pred3 =clf3.predict(X_test)

confusion = confusion_matrix(y_test3, y_pred3)
print(confusion)
fscore_aux = f1_score(y_test, y_pred, average='macro')
print('fscore: ', fscore_aux)
print('acabamos')