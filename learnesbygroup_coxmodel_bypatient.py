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
import pandas as pd

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
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

def fit_and_score_features(X, y):
    n_features = X.shape[1]
    scores = np.empty(n_features)
    m = CoxPHSurvivalAnalysis()
    for j in range(n_features):
        Xj = X[:, j:j+1]
        m.fit(Xj, y)
        scores[j] = m.score(Xj, y)
    return scores
"""PARTE DE TEST"""

names = ["QDA"]

classifiers = [

    QuadraticDiscriminantAnalysis()

]

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
    #topALT = np.array([74, 68, 67, 65, 62, 61, 58, 57, 50, 49, 48, 47, 45, 44])-1
   # topALT = np.array([28,29,82,86,90,92,102,109,168])-1
    topALT = np.array([157, 92, 143, 75, 71, 68, 130, 155, 136,150]) - 1
    #topALT = np.array([74,68,67,65,62,61,58,57,50,47,45,44,43,39,36,35,42,21,14,12,69,82,109,28,29,157]) - 1
    #topALT = np.array(
     #   [157,92,143,130,82,67,150,75,71,68]) - 1
    print(topALT)
    #train = Xtrain[:,topFeatIdx[0]]
    Xtrain = Xtrain[:,topALT]
    Groups = file_mat['Casos_n']
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


    Name_test = ["1Vs2"]
    ant = 0

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

        X = Xtrain
        y = y_train






        # iterate over classifiers
        score_KF=[]
        patients = []

        for name, clf in zip(names, classifiers):
          #  ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
            print(clf)
            print('training')
            logo = LeaveOneGroupOut()
            data_YY = []
            for train, test in logo.split(X, y, groups=Groups[:,0]):
                X_train, X_test = X[train], X[test]
                y_train, y_test = y[train], y[test]
                data = np.copy(X_train)
                data_y = y_test
                df = pd.DataFrame(data, columns=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])



                if data_y[0] == 0:
                    data_YY.append(365)
                else:
                    data_YY.append(365*2)

                data_y = np.array(data_y)






                #data_YY = [(data_y[r], 365) for r in range(data_y.__len__()) if data_y==]


#                 estimator = CoxPHSurvivalAnalysis()
#                 estimator.fit(data_x_numeric, data_YY)
#
#                 scores = fit_and_score_features(data_x_numeric.values, data_YY)
#                 #print(pd.Series(scores, index=data_x_numeric.columns).sort_values(ascending=False))
#
                X_test = X_test.mean(axis=0)
                patients.append(X_test)
#                # X_test = pd.DataFrame(X_test, columns=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
#                 pred_surv = estimator.predict_survival_function(X_test)
#                 print(pred_surv)
#                 for i, c in enumerate(pred_surv):
#                     plt.step(c.x, c.y, where="post", label="Sample %d" % (i + 1))
#                 plt.ylabel("est. probability of survival $\hat{S}(t)$")
#                 plt.xlabel("time $t$")
#                 plt.legend(loc="best")
#                 plt.show()
#                 #clf.fit(X_train, y_train)
# #                score = clf.score(X_test, y_test)
#             #scores = cross_val_score(clf, X_train, y_train, cv=5,scoring='f1_macro',error_score=np.nan)
#             #score = scores.mean()
#             #    print(name,': ',score>0.5)
#                 antt=pred_surv[0].y[0]>0.5
#                 ant=ant+antt
#                 print(antt)
#                 print(y_test[0])
#                 print('acc=',y_test[0]==antt)
                #y_pred = clf.predict(X_test[0].reshape((1,-1)))
            #    y_pred = clf.predict(X_test)
                #confusion = confusion_matrix(y_test,y_pred)
                #confusion = y_test[0]==y_pred[0]
                #ant=confusion+ant
                #print(ant)
           #     fscore_aux = f1_score(y_test, y_pred, average='macro')
            #    score_KF.append(score)
            #    print('score: ', fscore_aux)

        data_patient = pd.DataFrame(np.array(patients), columns=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
        data_YY = np.array(data_YY, dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
        from sksurv.preprocessing import OneHotEncoder
        from sksurv.linear_model import CoxPHSurvivalAnalysis



        data_x_numeric = OneHotEncoder().fit_transform(data_patient)
        data_x_numeric.head()
        estimator = CoxPHSurvivalAnalysis()
        estimator.fit(data_x_numeric, data_YY)
        scores = fit_and_score_features(data_x_numeric.values, data_YY)
        print(pd.Series(scores, index=data_x_numeric.columns).sort_values(ascending=False))
        import matplotlib.pyplot as plt
        from sksurv.nonparametric import kaplan_meier_estimator

        time, survival_prob = kaplan_meier_estimator(data_YY["Status"], data_YY["Survival_in_days"])
        plt.step(time, survival_prob, where="post")
        plt.ylabel("est. probability of survival $\hat{S}(t)$")
        plt.xlabel("time $t$")
        plt.show()
        print(estimator.score(data_x_numeric, data_YY))
        from sklearn.feature_selection import SelectKBest
        from sklearn.pipeline import Pipeline

        pipe = Pipeline([('encode', OneHotEncoder()),
                         ('select', SelectKBest(fit_and_score_features, k=3)),
                         ('model', CoxPHSurvivalAnalysis())])
        from sklearn.model_selection import GridSearchCV

        param_grid = {'select__k': np.arange(1, data_x_numeric.shape[1] + 1)}
        gcv = GridSearchCV(pipe, param_grid, return_train_score=True, cv=3, iid=True)
        gcv.fit(data_patient, data_YY)

        print(pd.DataFrame(gcv.cv_results_).sort_values(by='mean_test_score', ascending=False))
