#import pdb; pdb.st_trace()
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="2"
#os.environ["OMP_NUM_THREADS"] = "3"
#os.environ["MKL_NUM_THREADS"] = "19"
#os.environ["NUMEXPR_NUM_THREADS"] = "19"
#os.environ["OMP_NUM_THREADS"] = "19"

from keras import layers
from keras import models
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import numpy as np
import scipy.io
import numpy.matlib
import matplotlib.pyplot as plt
from PIL import Image as ImageOP
from os import system, name

import matplotlib.pyplot as plt
import glob
from keras.preprocessing import image
from itertools import product
from pathlib import Path
import os.path
import h5py
from sklearn.model_selection import train_test_split
#import os
##os.environ["CUDA_VISIBLE_DEVICES"]=""
#import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
#sys.path.append('/home/ricardo/PycharmProjects/pykernels-master/pykernels/')
#sys.path.append('/home/ricardo/PycharmProjects/pykernels-master/pykernels/graph/')

sys.path.append('pykernels-master/pykernels/')
sys.path.append('pykernels-master/pykernels/graph/')


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
from sklearn.metrics import roc_auc_score

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

names = [
    "Nearest Neighbors", "Linear SVM", "RBF SVM", "Chi2 SVM", "Cosine SVM", "GHK SVM", "Gaussian Process",
    "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
    "Naive Bayes",
    "QDA", "LogisticRegression", "GradientBoosting",
    #"Voting1vs3", "Voting1v2", "NewVoting",
    #"onevsallVoting"
    ]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.1,probability=True),
    SVC(gamma=2, C=1,probability=True),
    SVC(kernel=Chi2(), C=1,probability=True),
    SVC(kernel=Cossim(), C=1,probability=True),
    SVC(kernel=GeneralizedHistogramIntersection(), C=1,probability=True),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=10),
    RandomForestClassifier(max_depth=1, n_estimators=12, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(priors=None, var_smoothing=1e-04),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression(solver='lbfgs'),
    GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0),
    # VotingClassifier(estimators=[('QDA', QuadraticDiscriminantAnalysis()), ('NB', GaussianNB()),('Dtree', DecisionTreeClassifier(max_depth=5))], voting='hard')
    #VotingClassifier(estimators=[('LSVC', SVC(kernel="linear", C=0.025)),
    #                             ('GraBos', AdaBoostClassifier()),
     #                            ('NB', RandomForestClassifier(max_depth=5, n_estimators=300, max_features=2))],
      #               voting='hard'),

    #VotingClassifier(estimators=[('QDA', MLPClassifier(alpha=1)),
     #                            ('GraBos', GradientBoostingClassifier(n_estimators=300, learning_rate=1.0, max_depth=1,
      #                                                                 random_state=0)),
       #                          ('Lr', LogisticRegression(solver='lbfgs')),
        #                         ('Dtree', DecisionTreeClassifier(max_depth=10))], voting='hard'),

    #VotingClassifier(estimators=[('QDA', QuadraticDiscriminantAnalysis()),
      #                           ('GraBos', AdaBoostClassifier()),
     #                            ('Lr', MLPClassifier(alpha=1)),
       #                          ('Dtree', DecisionTreeClassifier(max_depth=10))], voting='hard'),

    #OneVsOneClassifier(GaussianNB())
]

#h5f = h5py.File(pathTestArray, 'r')
#XtestF = h5f['XtestAuto'][:]
#Ytest = h5f['Y_test'][:]
#h5f.close()
#print('Vector TEST LOAD!')


#files = glob.glob('/home/ricardo/Documentos/experimentosMay/')
#files=sorted(glob.glob('/home/ricardo/Documentos/SYNC/ExperimentosGrafoHaralickS/train*.mat'), key=os.path.basename)
files=sorted(glob.glob('/home/ricardo/Documents/Doctorado/CreacionArticulos/articuloPNWithAnne/borrar3/train*.mat'), key=os.path.basename)
#/home/moncayor/moncayoR/ExperimentosGrafoHaralickS

import scipy.io as sio
from sklearn.model_selection import cross_val_score
Kfold = 10
FileResults='prueba_grafo_Partition_haralickMeansV2Gooññd.npy'

if not(os.path.exists(FileResults)):
    ScoreAllFiles=[]
    for fileTrain in files:

        print(fileTrain)
        print(len(fileTrain))
        fileTR = fileTrain
#        fileTE = fileTrain[0:57]+'test'+fileTrain[62::]
        fileTE = fileTrain[0:79] + 'test' + fileTrain[79+5::]

        print(fileTE)

        file_mat=sio.loadmat(fileTR)
        Xtrain = file_mat['X']
        Ytrain = file_mat['Y']


        file_mat=sio.loadmat(fileTE)
        XtestF = file_mat['Xtest']
        Ytest = file_mat['Ytest']


        Ytrain = np.reshape(Ytrain,len(Ytrain))
        Ytest = np.reshape(Ytest,len(Ytest))

        Xtrain = Xtrain[:,:].astype(np.float32)
        XtestF = XtestF[:,:].astype(np.float32)


        Xtrain = Xtrain/4
        XtestF = XtestF/4
        portionX = np.int(Xtrain.shape[1]/21*5)
        Xtrain[:,portionX::] = Xtrain[:,portionX::]*2
        XtestF[:, portionX::] = XtestF[:, portionX::] * 2




        Y_train_X20 = Ytrain
        Y_test_X20 = Ytest



        """Find arrays"""
        index = Y_train_X20 == 1
        Xtrain1=Xtrain[index,:]
        Ytrain1 = Y_train_X20[index]
        Groups_train1 = [np.matlib.repmat(number,1,1) for number in range(0,len(index),1) if index[number]==True]
        Groups_train1 = np.matlib.reshape((Groups_train1), ( len(Groups_train1)*Groups_train1[0].shape[1],1))/Groups_train1[0].shape[1]

        index = Y_train_X20 == 2
        Xtrain2=Xtrain[index,:]
        Ytrain2 = Y_train_X20[index]
        Groups_train2 = [np.matlib.repmat(number,1,1) for number in range(0,len(index)) if index[number]==True]
        Groups_train2 = np.matlib.reshape((Groups_train2), ( len(Groups_train2)*1,1))+(Groups_train1[-1][0]-Groups_train2[0][0]+1)

        # index = Y_train_X20 == 2
        # Xtrain2=Xtrain[index]
        # Ytrain2 = Y_train_X20[index]

        index = Y_train_X20 == 3
        Xtrain3=Xtrain[index,:]
        Ytrain3 = Y_train_X20[index]
        Groups_train3 = [np.matlib.repmat(number,1,8) for number in range(0,len(index),8) if index[number]==True]
        Groups_train3 = ((np.matlib.reshape((Groups_train3), ( len(Groups_train3)*8,1))/8))
        Groups_train3 = Groups_train3 + (Groups_train2[-1][0]-Groups_train3[0][0]+1)
        #Groups_train1 = [np.matlib.repmat(number,1,8) for number in range(0,len(index),8) if index[number]==True]
        #Groups_train1 = np.matlib.reshape((Groups_train1), ( len(Groups_train1)*8,1))/8


        index = Y_test_X20 == 1
        Xtest1=XtestF[index,:]
        Ytest1 = Y_test_X20[index]
        index = Y_test_X20 == 2
        Xtest2=XtestF[index,:]
        Ytest2 = Y_test_X20[index]
        index = Y_test_X20 == 3
        Xtest3=XtestF[index,:]
        Ytest3 = Y_test_X20[index]

        Groups = np.vstack((Groups_train1,Groups_train2,Groups_train3))
        x = Xtrain
        y = Y_train_X20

        Kfold_n = 5
        indices1_K = np.random.permutation(np.int(len(Groups_train1)))
        indices2_K = np.random.permutation(np.int(len(Groups_train2)))+len(Groups_train1)
        indices3_K = np.random.permutation(np.int(len(Groups_train3) / 8))+len(Groups_train1)+len(Groups_train2)
        out_k1 = np.int(np.round(len(indices1_K)/Kfold_n))
        out_k2 = np.int(np.round(len(indices2_K) / Kfold_n))
        out_k3 = np.int(np.round(len(indices3_K) / Kfold_n))+1
        ScoreAll=[]
        for Kval in range(Kfold_n):
            print('Kfold:',Kval)
            Test_index1 = indices1_K[Kval*out_k1:(Kval+1)*out_k1]
            Train_index1 = np.setdiff1d(indices1_K,Test_index1)
            [IndexK1_train,noUsa]=np.where(Groups_train1==Train_index1)
            [IndexK1_test, noUsa] = np.where(Groups_train1 == Test_index1)
            IndexK1_test = IndexK1_test


            Test_index2 = indices2_K[Kval * out_k2:(Kval + 1) * out_k2]
            Train_index2 = np.setdiff1d(indices2_K, Test_index2)
            [IndexK2_train,noUsa]=np.where(Groups_train2==Train_index2)
            [IndexK2_test, noUsa] = np.where(Groups_train2 == Test_index2)
            IndexK2_test = IndexK2_test



            Test_index3 = indices3_K[Kval * out_k3:(Kval + 1) * out_k3]
            Train_index3 = np.setdiff1d(indices3_K, Test_index3)
            [IndexK3_train,noUsa]=np.where(Groups_train3==Train_index3)
            [IndexK3_test, noUsa] = np.where(Groups_train3 == Test_index3)
            IndexK3_test = IndexK3_test[0::8]

            Train_X1 = Xtrain1[IndexK1_train]
            Train_X2 = Xtrain2[IndexK2_train]
            Train_X3 = Xtrain3[IndexK3_train]
            Y_Ktrain1 = Ytrain1[IndexK1_train]
            Y_Ktrain2 = Ytrain2[IndexK2_train]
            Y_Ktrain3 = Ytrain3[IndexK3_train]
            print('Train',Train_X1.shape,Train_X2.shape,Train_X3.shape)
            Test_X1 = Xtrain1[IndexK1_test]
            Test_X2 = Xtrain2[IndexK2_test]
            Test_X3 = Xtrain3[IndexK3_test]
            Y_Ktest1 = Ytrain1[IndexK1_test]
            Y_Ktest2 = Ytrain2[IndexK2_test]
            Y_Ktest3 = Ytrain3[IndexK3_test]
            print('Test:',Test_X1.shape, Test_X2.shape, Test_X3.shape)


    ##Reduce number of class 2

        #    clus = 80
        #    kmeans = KMeans(n_clusters=clus, random_state=0,max_iter=1000).fit(Train_X2)
        #    Ytrain2K=Y_Ktrain2[0:clus]
        #    kmeans =kmeans.cluster_centers_
 
            kmeans = Train_X2
            Ytrain2K = Y_Ktrain2

#            Name_test = ["3VS.1&2","1VS.2&3"
#                ,"2VS.1&3"]

            Name_test = ["3VS.1&2","1VS.2&3"
                ,"1Vs2","1Vs3","2Vs3","2VS.1&3","1Vs2AllT","1Vs3AllT","2Vs3AllT"
                         ]
            ScoreKval = []
            for z in range(len(Name_test)):
                print(Name_test[z])
                #3vs1&2
                if z==0:
                    Xtotal = np.concatenate((Train_X1, kmeans, Train_X3), axis=0)
                    Ytotal = np.concatenate((Y_Ktrain1*0, Ytrain2K * 0, Y_Ktrain3/3), axis=0)
                    Xtesttotal = np.concatenate((Test_X1, Test_X2, Test_X3), axis=0)
                    Ytesttotal = np.concatenate((Y_Ktest1*0, Y_Ktest2 * 0, Y_Ktest3/3 ), axis=0)

                    Xtesttotal_Dtest = np.concatenate((Xtest1, Xtest2, Xtest3), axis=0)
                    Ytesttotal_Dtest = np.concatenate((Ytest1 * 0, Ytest2 * 0, Ytest3 / 3), axis=0)
                #1vs2&3
                elif z == 1:
                    Xtotal = np.concatenate((Train_X1, kmeans, Train_X3), axis=0)
                    Ytotal = np.concatenate((Y_Ktrain1 , Ytrain2K * 0, Y_Ktrain3*0), axis=0)
                    Xtesttotal = np.concatenate((Test_X1, Test_X2, Test_X3), axis=0)
                    Ytesttotal = np.concatenate((Y_Ktest1 , Y_Ktest2 * 0, Y_Ktest3 *0), axis=0)

                    Xtesttotal_Dtest = np.concatenate((Xtest1, Xtest2, Xtest3), axis=0)
                    Ytesttotal_Dtest = np.concatenate((Ytest1 , Ytest2 * 0, Ytest3 *0), axis=0)
                 #1vs2
                elif z==2:
                    Xtotal = np.concatenate((Train_X1, kmeans), axis=0)
                    Ytotal = np.concatenate((Y_Ktrain1 , Ytrain2K * 0), axis=0)
                    Xtesttotal = np.concatenate((Test_X1, Test_X2), axis=0)
                    Ytesttotal = np.concatenate((Y_Ktest1 , Y_Ktest2 * 0), axis=0)
                    Xtesttotal_Dtest = np.concatenate((Xtest1, Xtest2), axis=0)
                    Ytesttotal_Dtest = np.concatenate((Ytest1 ,Ytest2*0), axis=0)
                 #1vs3
                elif z==3:
                    Xtotal = np.concatenate((Train_X1, Train_X3), axis=0)
                    Ytotal = np.concatenate((Y_Ktrain1 , Y_Ktrain3 *0), axis=0)
                    Xtesttotal = np.concatenate((Test_X1, Test_X3), axis=0)
                    Ytesttotal = np.concatenate((Y_Ktest1, Y_Ktest3 *0), axis=0)

                    Xtesttotal_Dtest = np.concatenate((Xtest1,  Xtest3), axis=0)
                    Ytesttotal_Dtest = np.concatenate((Ytest1, Ytest3 *0), axis=0)
                #2vs3
                elif z == 4:
                    Xtotal = np.concatenate(( kmeans, Train_X3), axis=0)
                    Ytotal = np.concatenate(( Ytrain2K/2 , Y_Ktrain3 *0), axis=0)
                    Xtesttotal = np.concatenate(( Test_X2, Test_X3), axis=0)
                    Ytesttotal = np.concatenate(( Y_Ktest2/2, Y_Ktest3*0), axis=0)

                    Xtesttotal_Dtest = np.concatenate(( Xtest2, Xtest3), axis=0)
                    Ytesttotal_Dtest = np.concatenate((Ytest2/2, Ytest3*0), axis=0)



               #2vs1&3     
                elif z == 5:

                    Xtotal = np.concatenate((Train_X1, kmeans, Train_X3), axis=0)
                    Ytotal = np.concatenate((Y_Ktrain1 * 0, Ytrain2K/2, Y_Ktrain3*0), axis=0)
                    Xtesttotal = np.concatenate((Test_X1, Test_X2, Test_X3), axis=0)
                    Ytesttotal = np.concatenate((Y_Ktest1 * 0, Y_Ktest2/2, Y_Ktest3*0), axis=0)


                    Xtesttotal_Dtest = np.concatenate((Xtest1, Xtest2, Xtest3), axis=0)
                    Ytesttotal_Dtest = np.concatenate((Ytest1 * 0, Ytest2/2, Ytest3*0), axis=0)
## PARA LAS PRUEBAS DE VALIDACION
                #1 VS 3
                if z==6:
                    Xtotal = np.concatenate((Train_X1, Train_X3), axis=0)
                    Ytotal = np.concatenate((Y_Ktrain1 , Y_Ktrain3), axis=0)
                    Xtesttotal = np.concatenate((Test_X1, Test_X2, Test_X3), axis=0)
                    Ytesttotal = np.concatenate((Y_Ktest1, Y_Ktest2 , Y_Ktest3 ), axis=0)

                    Xtesttotal_Dtest = np.concatenate((Xtest1, Xtest2, Xtest3), axis=0)
                    Ytesttotal_Dtest = np.concatenate((Ytest1 , Ytest2 , Ytest3 ), axis=0)
                #1 vS 2
                if z == 7:
                    Xtotal = np.concatenate((Train_X1, kmeans), axis=0)
                    Ytotal = np.concatenate((Y_Ktrain1 , Ytrain2K ), axis=0)
                    Xtesttotal = np.concatenate((Test_X1, Test_X2, Test_X3), axis=0)
                    Ytesttotal = np.concatenate((Y_Ktest1 , Y_Ktest2 , Y_Ktest3 ), axis=0)

                    Xtesttotal_Dtest = np.concatenate((Xtest1, Xtest2, Xtest3), axis=0)
                    Ytesttotal_Dtest = np.concatenate((Ytest1 , Ytest2, Ytest3 ), axis=0)

                # 2 vS 3
                if z == 8:
                    Xtotal = np.concatenate(( kmeans, Train_X3), axis=0)
                    Ytotal = np.concatenate((Ytrain2K , Y_Ktrain3 ), axis=0)

                    Xtesttotal = np.concatenate((Test_X1, Test_X2, Test_X3), axis=0)
                    Ytesttotal = np.concatenate((Y_Ktest1 , Y_Ktest2 , Y_Ktest3 ), axis=0)
                    Xtesttotal_Dtest = np.concatenate((Xtest1, Xtest2, Xtest3), axis=0)
                    Ytesttotal_Dtest = np.concatenate((Ytest1 , Ytest2 , Ytest3 ), axis=0)



                X_train = Xtotal
                X_test = Xtesttotal
                y_train = Ytotal
                y_test = Ytesttotal

                X_test_Dtest = Xtesttotal_Dtest
                y_test_Dtest = Ytesttotal_Dtest
              #  X_train = preprocessing.MinMaxScaler().fit_transform(Xtotal)
              #  X_test = preprocessing.MinMaxScaler().fit_transform(Xtesttotal)
              #  X_test_Dtest = preprocessing.MinMaxScaler().fit_transform(Xtesttotal_Dtest)

                X_train = preprocessing.scale(X_train)
                X_test = preprocessing.scale(X_test)
                X_test_Dtest = preprocessing.scale(X_test_Dtest)
                #X_train = preprocessing.MinMaxScaler().fit_transform(Xtotal)
                #X_test = preprocessing.MinMaxScaler().fit_transform(Xtesttotal)
                y_train = Ytotal
                y_test = Ytesttotal
                y_test_Dtest = Ytesttotal_Dtest




                # iterate over classifiers
                score_KF=[]
                for name, clf in zip(names, classifiers):
                  #  ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
                    print('Task',name)
               #     print(clf)
                    clf.fit(X_train, y_train)
                    #score = clf.score(X_test, y_test,scoring='f1_macro')
                    #scores = cross_val_score(clf, X_train, y_train, cv=5,scoring='f1_macro')
                    #score = scores.mean()
                    #print(name,': ',score)
                    y_pred = clf.predict(X_test)
                    y_pred_scores = clf.predict_proba(X_test)
                    y_pred_Dtest = clf.predict(X_test_Dtest )
                    y_pred_Dtest_scores = clf.predict_proba(X_test_Dtest)
                    confusion = confusion_matrix(y_test,y_pred)
                    print('train:\n',confusion)
                    confusionDtest = confusion_matrix(Ytesttotal_Dtest, y_pred_Dtest)
                    print('test:\n',confusionDtest)
                    fscore_aux = f1_score(y_test, y_pred, average='macro')
                    fscore_auxTest = f1_score(Ytesttotal_Dtest, y_pred_Dtest, average='macro')
                    print(y_pred_scores.shape[1])
                    if np.int(np.unique(y_test).shape[0])<=2:
                        RocTrain = roc_auc_score(y_test, y_pred_scores[:,1])
                    #RocTrain = roc_auc_score(y_test, y_pred_scores)
                        RocTest = roc_auc_score(Ytesttotal_Dtest, y_pred_Dtest_scores[:,1])
                    else:
                        RocTrain = 0
                        RocTest = 0

#                    score_KF.append((Name_test[z],name,fscore_aux,fscore_auxTest,y_pred_scores,y_pred_Dtest_scores,confusion,confusionDtest))
                    score_KF.append((Name_test[z],name,fscore_aux,fscore_auxTest,y_test,y_pred_scores,Ytesttotal_Dtest,y_pred_Dtest_scores,confusion,confusionDtest,RocTrain,RocTest))

                    print('score: ', fscore_aux, fscore_auxTest)
                    print('AUC:',RocTrain,RocTest)
                ScoreKval.extend(score_KF)
            ScoreAll.append((Kval,ScoreKval))
        ScoreAllFiles.append((fileTR,ScoreAll))
    np.save(FileResults, ScoreAllFiles)
ScoreAllFiles = np.load(FileResults,allow_pickle=True)
print('Y ahora el orden...')
FscoreK_train_files = []
FscoreK_test_files = []
for nfile in range(0,len(ScoreAllFiles)):
    print('Parametros: ',ScoreAllFiles[np.int(nfile)][0])
    FscoreK_train = []
    FscoreK_test = []

    for kn_val in range(0,5):
        list_Experim = [list(i) for i in ScoreAllFiles[np.int(nfile)][1][np.int(kn_val)][1]]
        Rows_train = np.array([])
        Rows_test = np.array([])
        name_clas = []
        name_prueba = []
        for experimento in range(len(list_Experim)):
            Rows_train = np.concatenate([Rows_train,[list_Experim[experimento][2]]])
            Rows_test= np.concatenate([Rows_test,[list_Experim[experimento][3]]])
            name_clas = np.concatenate([name_clas,[list_Experim[experimento][1]]])
            name_prueba = np.concatenate([name_prueba, [list_Experim[experimento][0]]])
        FscoreK_train.append(Rows_train)
        FscoreK_test.append(Rows_test)
        #convertir a arrays
    FscoreK_test = np.array(FscoreK_test)
    FscoreK_test_mean =np.mean(np.array(FscoreK_test), axis=0)
    FscoreK_train = np.array(FscoreK_train)
    FscoreK_train_mean = np.mean(np.array(FscoreK_train), axis=0)
    total_train = np.concatenate([[name_clas],[name_prueba], [FscoreK_train_mean]])
    total_test = np.concatenate([[name_clas], [name_prueba], [FscoreK_test_mean]])
    FscoreK_train_files.append(FscoreK_train_mean)
    FscoreK_test_files.append(FscoreK_test_mean)
all_train_val = np.array(FscoreK_train_files)
all_test_val = np.array(FscoreK_test_files)

indx_max = np.argmax(all_train_val,axis=0)
val_maxT = all_train_val[indx_max]
val_maxTest = all_test_val[indx_max]

lista_train = np.vstack((name_clas,name_prueba,val_maxT.diagonal(),val_maxTest.diagonal()))
indx_max = np.argmax(all_test_val,axis=0)
val_maxT = all_train_val[indx_max]
val_maxTest = all_test_val[indx_max]
lista_test = np.vstack((name_clas,name_prueba,val_maxT.diagonal(),val_maxTest.diagonal()))
#sacar mean y std de test y train
print('Acabo la evaluacion')
Sel_best = [1,19,80]

print(fileTrain)
fileTR = fileTrain
fileTE = fileTrain[0:50] + 'test' + fileTrain[55::]
print(fileTE)

file_mat=sio.loadmat(fileTR)
Xtrain = file_mat['X']
Ytrain = file_mat['Y']


file_mat=sio.loadmat(fileTE)
XtestF = file_mat['Xtest']
Ytest = file_mat['Ytest']


Ytrain = np.reshape(Ytrain,len(Ytrain))
Ytest = np.reshape(Ytest,len(Ytest))

Xtrain = Xtrain[:,:].astype(np.float32)
XtestF = XtestF[:,:].astype(np.float32)


Y_train_X20 = Ytrain
Y_test_X20 = Ytest



"""Find arrays"""
index = Y_train_X20 == 1
Xtrain1=Xtrain[index,:]
Ytrain1 = Y_train_X20[index]
Groups_train1 = [np.matlib.repmat(number,1,8) for number in range(0,len(index),8) if index[number]==True]
Groups_train1 = np.matlib.reshape((Groups_train1), ( len(Groups_train1)*8,1))/8

index = Y_train_X20 == 2
Xtrain2=Xtrain[index,:]
Ytrain2 = Y_train_X20[index]
Groups_train2 = [np.matlib.repmat(number,1,1) for number in range(0,len(index)) if index[number]==True]
Groups_train2 = np.matlib.reshape((Groups_train2), ( len(Groups_train2)*1,1))-161

# index = Y_train_X20 == 2
# Xtrain2=Xtrain[index]
# Ytrain2 = Y_train_X20[index]

index = Y_train_X20 == 3
Xtrain3=Xtrain[index,:]
Ytrain3 = Y_train_X20[index]
Groups_train3 = [np.matlib.repmat(number,1,4) for number in range(0,len(index),4) if index[number]==True]
Groups_train3 = (np.matlib.reshape((Groups_train3), ( len(Groups_train3)*4,1)))/4-19+164


index = Y_test_X20 == 1
Xtest1=XtestF[index,:]
Ytest1 = Y_test_X20[index]
index = Y_test_X20 == 2
Xtest2=XtestF[index,:]
Ytest2 = Y_test_X20[index]
index = Y_test_X20 == 3
Xtest3=XtestF[index,:]
Ytest3 = Y_test_X20[index]

Groups = np.vstack((Groups_train1,Groups_train2,Groups_train3))
x = Xtrain
y = Y_train_X20

Kfold_n = 5
indices1_K = np.random.permutation(np.int(len(Groups_train1)/8))
indices2_K = np.random.permutation(np.int(len(Groups_train2)))+23
indices3_K = np.random.permutation(np.int(len(Groups_train3) / 4))+225+23

out_k1 = np.int(np.round(len(indices1_K)/Kfold_n))
out_k2 = np.int(np.round(len(indices2_K) / Kfold_n))
out_k3 = np.int(np.round(len(indices3_K) / Kfold_n))+1
ScoreAll=[]
for Kval in range(Kfold_n):
    print('Kfold:',Kval)
    Test_index1 = indices1_K[Kval*out_k1:(Kval+1)*out_k1]
    Train_index1 = np.setdiff1d(indices1_K,Test_index1)
    [IndexK1_train,noUsa]=np.where(Groups_train1==Train_index1)
    [IndexK1_test, noUsa] = np.where(Groups_train1 == Test_index1)
    IndexK1_test = IndexK1_test[0::8]


    Test_index2 = indices2_K[Kval * out_k2:(Kval + 1) * out_k2]
    Train_index2 = np.setdiff1d(indices2_K, Test_index2)
    [IndexK2_train,noUsa]=np.where(Groups_train2==Train_index2)
    [IndexK2_test, noUsa] = np.where(Groups_train2 == Test_index2)
    IndexK2_test = IndexK2_test



    Test_index3 = indices3_K[Kval * out_k3:(Kval + 1) * out_k3]
    Train_index3 = np.setdiff1d(indices3_K, Test_index3)
    [IndexK3_train,noUsa]=np.where(Groups_train3==Train_index3)
    [IndexK3_test, noUsa] = np.where(Groups_train3 == Test_index3)
    IndexK3_test = IndexK3_test[0::4]

    Train_X1 = Xtrain1[IndexK1_train]
    Train_X2 = Xtrain2[IndexK2_train]
    Train_X3 = Xtrain3[IndexK3_train]
    Y_Ktrain1 = Ytrain1[IndexK1_train]
    Y_Ktrain2 = Ytrain2[IndexK2_train]
    Y_Ktrain3 = Ytrain3[IndexK3_train]
    print('Train',Train_X1.shape,Train_X2.shape,Train_X3.shape)
    Test_X1 = Xtrain1[IndexK1_test]
    Test_X2 = Xtrain2[IndexK2_test]
    Test_X3 = Xtrain3[IndexK3_test]
    Y_Ktest1 = Ytrain1[IndexK1_test]
    Y_Ktest2 = Ytrain2[IndexK2_test]
    Y_Ktest3 = Ytrain3[IndexK3_test]
    print('Test:',Test_X1.shape, Test_X2.shape, Test_X3.shape)


##Reduce number of class 2

    clus = 80
    kmeans = KMeans(n_clusters=clus, random_state=0,max_iter=1000).fit(Train_X2)
    Ytrain2K=Y_Ktrain2[0:clus]
    kmeans =kmeans.cluster_centers_



Name_test = ["3VS.1&2","1VS.2&3"
    ,"1Vs2","1Vs3","2Vs3","2VS.1&3"
             ]
ScoreKval = []
for z in range(len(Name_test)):
    print(Name_test[z])
    if z==0:
        Xtotal = np.concatenate((Train_X1, kmeans, Train_X3), axis=0)
        Ytotal = np.concatenate((Y_Ktrain1*0, Ytrain2K * 0, Y_Ktrain3/3), axis=0)
        Xtesttotal = np.concatenate((Test_X1, Test_X2, Test_X3), axis=0)
        Ytesttotal = np.concatenate((Y_Ktest1*0, Y_Ktest2 * 0, Y_Ktest3/3 ), axis=0)

        Xtesttotal_Dtest = np.concatenate((Xtest1, Xtest2, Xtest3), axis=0)
        Ytesttotal_Dtest = np.concatenate((Ytest1 * 0, Ytest2 * 0, Ytest3 / 3), axis=0)

    elif z == 1:
        Xtotal = np.concatenate((Train_X1, kmeans, Train_X3), axis=0)
        Ytotal = np.concatenate((Y_Ktrain1 , Ytrain2K * 0, Y_Ktrain3*0), axis=0)
        Xtesttotal = np.concatenate((Test_X1, Test_X2, Test_X3), axis=0)
        Ytesttotal = np.concatenate((Y_Ktest1 , Y_Ktest2 * 0, Y_Ktest3 *0), axis=0)

        Xtesttotal_Dtest = np.concatenate((Xtest1, Xtest2, Xtest3), axis=0)
        Ytesttotal_Dtest = np.concatenate((Ytest1 , Ytest2 * 0, Ytest3 *0), axis=0)

    elif z==2:
        Xtotal = np.concatenate((Train_X1, kmeans), axis=0)
        Ytotal = np.concatenate((Y_Ktrain1 , Ytrain2K * 0), axis=0)
        Xtesttotal = np.concatenate((Test_X1, Test_X2), axis=0)
        Ytesttotal = np.concatenate((Y_Ktest1 , Y_Ktest2 * 0), axis=0)

        Xtesttotal_Dtest = np.concatenate((Xtest1, Xtest2), axis=0)
        Ytesttotal_Dtest = np.concatenate((Ytest1 ,Ytest2*0), axis=0)

    elif z==3:
        Xtotal = np.concatenate((Train_X1, Train_X3), axis=0)
        Ytotal = np.concatenate((Y_Ktrain1 , Y_Ktrain3 *0), axis=0)
        Xtesttotal = np.concatenate((Test_X1, Test_X3), axis=0)
        Ytesttotal = np.concatenate((Y_Ktest1, Y_Ktest3 *0), axis=0)

        Xtesttotal_Dtest = np.concatenate((Xtest1,  Xtest3), axis=0)
        Ytesttotal_Dtest = np.concatenate((Ytest1, Ytest3 *0), axis=0)
    elif z == 4:
        Xtotal = np.concatenate(( kmeans, Train_X3), axis=0)
        Ytotal = np.concatenate(( Ytrain2K/2 , Y_Ktrain3 *0), axis=0)
        Xtesttotal = np.concatenate(( Test_X2, Test_X3), axis=0)
        Ytesttotal = np.concatenate(( Y_Ktest2/2, Y_Ktest3*0), axis=0)

        Xtesttotal_Dtest = np.concatenate(( Xtest2, Xtest3), axis=0)
        Ytesttotal_Dtest = np.concatenate((Ytest2/2, Ytest3*0), axis=0)




    elif z == 5:

        Xtotal = np.concatenate((Train_X1, kmeans, Train_X3), axis=0)
        Ytotal = np.concatenate((Y_Ktrain1 * 0, Ytrain2K/2, Y_Ktrain3*0), axis=0)
        Xtesttotal = np.concatenate((Test_X1, Test_X2, Test_X3), axis=0)
        Ytesttotal = np.concatenate((Y_Ktest1 * 0, Y_Ktest2/2, Y_Ktest3*0), axis=0)


        Xtesttotal_Dtest = np.concatenate((Xtest1, Xtest2, Xtest3), axis=0)
        Ytesttotal_Dtest = np.concatenate((Ytest1 * 0, Ytest2/2, Ytest3*0), axis=0)




    X_train = Xtotal
    X_test = Xtesttotal
    y_train = Ytotal
    y_test = Ytesttotal

    X_test_Dtest = Xtesttotal_Dtest
    y_test_Dtest = Ytesttotal_Dtest
  #  X_train = preprocessing.MinMaxScaler().fit_transform(Xtotal)
  #  X_test = preprocessing.MinMaxScaler().fit_transform(Xtesttotal)
  #  X_test_Dtest = preprocessing.MinMaxScaler().fit_transform(Xtesttotal_Dtest)

    X_train = preprocessing.scale(X_train)
    X_test = preprocessing.scale(X_test)
    X_test_Dtest = preprocessing.scale(X_test_Dtest)
    #X_train = preprocessing.MinMaxScaler().fit_transform(Xtotal)
    #X_test = preprocessing.MinMaxScaler().fit_transform(Xtesttotal)
    y_train = Ytotal
    y_test = Ytesttotal
    y_test_Dtest = Ytesttotal_Dtest




    # iterate over classifiers
    score_KF=[]
    for name, clf in zip(names, classifiers):
      #  ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        print('Task',name)
   #     print(clf)
        clf.fit(X_train, y_train)
        #score = clf.score(X_test, y_test,scoring='f1_macro')
        #scores = cross_val_score(clf, X_train, y_train, cv=5,scoring='f1_macro')
        #score = scores.mean()
        #print(name,': ',score)
        y_pred = clf.predict(X_test)
        y_pred_Dtest = clf.predict(X_test_Dtest )
        confusion = confusion_matrix(y_test,y_pred)
        print(confusion)
        fscore_aux = f1_score(y_test, y_pred, average='macro')
        fscore_auxTest = f1_score(Ytesttotal_Dtest, y_pred_Dtest, average='macro')
        score_KF.append((Name_test[z],name,fscore_aux,fscore_auxTest))
        print('score: ', fscore_aux, fscore_auxTest)

#armar la lista del archivo




            #
            # indice = score_KF.index(max(score_KF))
            # print(Name_test[z],'Best Classifier:\n',names[indice],'\nScore KFOLD: ', score_KF[indice])
            # print('FINAL RESULT: ',Name_test[z])
            # clf = classifiers[indice].fit(X_train, y_train)
            # score = clf.score(X_test, y_test)
            # print(names[indice],'Test SET score: ', score)
            # y_pred = clf.predict(X_test)
            # confusion = confusion_matrix(y_test, y_pred)
            # print(confusion)
            # print('F-score Test:',f1_score(y_test, y_pred, average='macro'))
            #
            #
            #
            #
            #
            #
            #



































