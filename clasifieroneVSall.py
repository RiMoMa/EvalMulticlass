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
from pykernels.regular import Chi2,Cossim,GeneralizedHistogramIntersection


import pickle

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



#file_mat=sio.loadmat('/media/ricardo/My Passport/internship/ArraysSB/server50WSall/trainNoNormSB_Words50block_48L1_F.mat')
file_mat=sio.loadmat('/home/ricardo/Documentos/SYNC/ExperimentosGrafoHaralickS/trainNorm_augmented_Words50block_48L2_END.mat')
Xtrain = file_mat['X']
Ytrain = file_mat['Y']
file_mat=sio.loadmat('/home/ricardo/Documentos/SYNC/ExperimentosGrafoHaralickS/testNorm_augmented_Words50block_48L2_END.mat')
XtestF = file_mat['Xtest']
Ytest = file_mat['Ytest']
Ytrain = np.reshape(Ytrain,len(Ytrain))
Ytest = np.reshape(Ytest,len(Ytest))

# definir los niveles a usar
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

##Definir el experimento
Xtotal = np.concatenate((Xtrain1, kmeans, Xtrain3), axis=0)
Ytotal = np.concatenate((Ytrain1 , Ytrain2K * 0, Ytrain3*0), axis=0)
Xtesttotal = np.concatenate((Xtest1, Xtest2, Xtest3), axis=0)
Ytesttotal = np.concatenate((Ytest1 , Ytest2*0 , Ytest3*0), axis=0)
YtestCompleto = np.concatenate((Ytest1 , Ytest2 , Ytest3), axis=0)
X_train = Xtotal
X_test = Xtesttotal
y_train = Ytotal
y_test = Ytesttotal
#normalizacion
X_train = preprocessing.MinMaxScaler().fit_transform(Xtotal)
X_test = preprocessing.MinMaxScaler().fit_transform(Xtesttotal)
X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)


import pandas as pd

import pymrmr
from sklearn.preprocessing import KBinsDiscretizer
y_trainST = np.reshape(y_train,(len(y_train),1))
# est = KBinsDiscretizer(n_bins=10)
# Xtotal2=np.ndarray.tolist(Xtotal)
# # est.fit(Xtotal2)
# #Xt = est.transform(Xtotal2)
# Xt=np.concatenate((y_trainST,Xtotal),axis=1)
# df=pd.DataFrame(data=Xt[0:,0:],
#                 index=[i for i in range(Xt.shape[0])],
#                 columns=['f'+str(i) for i in range(Xt.shape[1])])
# index = pymrmr.mRMR(df, 'MIQ',20)
# #otra forma mrmr
from skfeature.function.information_theoretical_based import MRMR
Xtotal = Xtotal.astype(float)
idx, _, _ = MRMR.mrmr(Xtotal, y_train, n_selected_features=1000)


from feast import mRMR

sf = mRMR(data, labels, n_select)
# obtain the dataset on the selected features
X_train = X_train[:, idx[0:1000]]


#X_train = df[index].as_matrix()

clf =SVC(kernel=GeneralizedHistogramIntersection(), C=1,probability=True)
clf.fit(X_train, y_train)
# df=pd.DataFrame(data=X_test[0:,0:],
#                 index=[i for i in range(X_test.shape[0])],
#                 columns=['f'+str(i) for i in range(X_test.shape[1])])
# X_test = df[index].as_matrix()

score = clf.score(X_test[:, idx[0:1000]], y_test)
print('3Vs12.',': ',score)
y_pred = clf.predict_proba(X_test[:, idx[0:1000]])




file_mat=sio.loadmat('/home/ricardo/Documentos/SYNC/ExperimentosGrafoHaralickS/trainNorm_augmented_Words300block_24L3_END.mat')
Xtrain = file_mat['X']
Ytrain = file_mat['Y']
file_mat=sio.loadmat('/home/ricardo/Documentos/SYNC/ExperimentosGrafoHaralickS/testNorm_augmented_Words300block_24L3_END.mat')
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
Ytotal = np.concatenate((Ytrain1*0 , Ytrain2K , Ytrain3*0), axis=0)
Xtesttotal = np.concatenate((Xtest1, Xtest2, Xtest3), axis=0)
Ytesttotal = np.concatenate((Ytest1*0 , Ytest2 , Ytest3*0), axis=0)
YtestCompleto = np.concatenate((Ytest1 , Ytest2 , Ytest3), axis=0)



X_train = Xtotal
X_test = Xtesttotal
y_train = Ytotal
y_test = Ytesttotal
X_train = preprocessing.MinMaxScaler().fit_transform(Xtotal)
X_test = preprocessing.MinMaxScaler().fit_transform(Xtesttotal)
X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)




#
#
# df=pd.DataFrame(data=X_train[0:,0:],
#                 index=[i for i in range(X_train.shape[0])],
#                 columns=['f'+str(i) for i in range(X_train.shape[1])])
# index = pymrmr.mRMR(df, 'MIQ', 70)
#
# X_train = df[index].as_matrix()


# df=pd.DataFrame(data=X_test[0:,0:],
#                 index=[i for i in range(X_test.shape[0])],
#                 columns=['f'+str(i) for i in range(X_test.shape[1])])
# X_test = df[index].as_matrix()

Xtotal = Xtotal.astype(float)
idx, _, _ = MRMR.mrmr(Xtotal, y_train, n_selected_features=1000)

# obtain the dataset on the selected features
X_train = X_train[:, idx[0:1000]]

clf2 = AdaBoostClassifier()

clf2.fit(X_train, y_train)
X_test2 = X_test
y_pred2 =clf2.predict_proba(X_test2[:, idx[0:1000]])




file_mat=sio.loadmat('/home/ricardo/Documentos/SYNC/ExperimentosGrafoHaralickS/trainNorm_augmented_Words50block_48L0_END.mat')
Xtrain = file_mat['X']
Ytrain = file_mat['Y']
file_mat=sio.loadmat('/home/ricardo/Documentos/SYNC/ExperimentosGrafoHaralickS/testNorm_augmented_Words50block_48L0_END.mat')
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
Ytotal = np.concatenate((Ytrain1*0 , Ytrain2K * 0, Ytrain3), axis=0)
Xtesttotal = np.concatenate((Xtest1, Xtest2, Xtest3), axis=0)
Ytesttotal = np.concatenate((Ytest1*0 , Ytest2*0 , Ytest3), axis=0)
YtestCompleto = np.concatenate((Ytest1 , Ytest2 , Ytest3), axis=0)

X_train = Xtotal
X_test = Xtesttotal
y_train = Ytotal
y_test = Ytesttotal
X_train = preprocessing.MinMaxScaler().fit_transform(Xtotal)
X_test = preprocessing.MinMaxScaler().fit_transform(Xtesttotal)
X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)




# df=pd.DataFrame(data=X_train[0:,0:],
#                 index=[i for i in range(X_train.shape[0])],
#                 columns=['f'+str(i) for i in range(X_train.shape[1])])
# index = pymrmr.mRMR(df, 'MIQ', 70)
#
# X_train = df[index].as_matrix()
Xtotal = Xtotal.astype(float)
idx, _, _ = MRMR.mrmr(Xtotal, y_train, n_selected_features=1000)

# obtain the dataset on the selected features
X_train = X_train[:, idx[0:1000]]

clf3 =  MLPClassifier(alpha=1)

clf3.fit(X_train, y_train)
y_pred3 =clf3.predict_proba(X_test[:, idx[0:1000]])

y_pred1=y_pred[:,1]
y_pred2l=y_pred2[:,1]
y_pred3l=y_pred3[:,1]
y_predall = np.vstack((y_pred1,y_pred2l,y_pred3l))
Y_predall=np.argmax(y_predall,axis=0)+1
confusion = confusion_matrix(YtestCompleto,Y_predall)
print(confusion)
fscore_aux = f1_score(YtestCompleto, Y_predall, average='macro')
print(fscore_aux)