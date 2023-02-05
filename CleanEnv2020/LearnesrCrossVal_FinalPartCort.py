#import pdb; pdb.set_trace()
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="2"
#os.environ["OMP_NUM_THREADS"] = "3"
#os.environ["MKL_NUM_THREADS"] = "19"
#os.environ["NUMEXPR_NUM_THREADS"] = "19"
#os.environ["OMP_NUM_THREADS"] = "19"
import pickle
import pandas as pd

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
from sklearn.metrics import fbeta_score
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
from imblearn.ensemble import RUSBoostClassifier

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
    "Nearest Neighbors",
    #"Linear SVM",
    #"RBF SVM",
    #"Chi2 SVM",
    #"Cosine SVM",
    #"GHK SVM",
    #"Gaussian Process",
  #  "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
    "LogisticRegression",
    "GradientBoosting",
    "RUSBOOST"
    #"Voting1vs3", "Voting1v2", "NewVoting",
    #"onevsallVoting"
    ]

classifiers = [
    KNeighborsClassifier(3),
    #SVC(kernel="linear", C=0.1,probability=True),
    #SVC(gamma=2, C=1,probability=True),
    #SVC(kernel=Chi2(), C=1,probability=True),
    #SVC(kernel=Cossim(), C=1,probability=True),
    #SVC(kernel=GeneralizedHistogramIntersection(), C=1,probability=True),
    #GaussianProcessClassifier(1.0 * RBF(1.0)),
    #DecisionTreeClassifier(max_depth=10),
    RandomForestClassifier(max_depth=1, n_estimators=50, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(priors=None, var_smoothing=1e-04),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression(solver='lbfgs'),
    GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0),
    RUSBoostClassifier(random_state=0),
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
files=sorted(glob.glob('/mnt/md0/ricardo/PruebaKfoldC/train*.mat'), key=os.path.basename)
#/home/moncayor/moncayoR/ExperimentosGrafoHaralickS

import scipy.io as sio
from sklearn.model_selection import cross_val_score
Kfold = 10
FileResults='/mnt/md0/ricardo/PruebaKfoldC/ResultadobF5.pkl'

Experimento = []
KfoldP = []

Metrica = []

if not(os.path.exists(FileResults)):
    ScoreAllKvalues=[]


    for fileTrain in files:

        print(fileTrain)
        print(len(fileTrain))
        fileTR = fileTrain
        fileTE = fileTrain[0:30] + 'test' + fileTrain[35::]
        Experimento.append(fileTrain[fileTrain.find('Words'):fileTrain.find('_Kfold')])
        KfoldP.append(fileTrain[fileTrain.find('_Kfold'):fileTrain.find('.mat')])
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
        Xtrain[:,::] = Xtrain[:,:]*(1/4)
        XtestF[:,::] = XtestF[:, :] *(1/4)

        min_max_scaler = preprocessing.MinMaxScaler()
        Xtrain = min_max_scaler.fit_transform(Xtrain)
        XtestF = min_max_scaler.transform(XtestF)




        Y_train_X20 = Ytrain
        Y_test_X20 = Ytest



        """Find arrays"""
        index = Y_train_X20 == 1
        Xtrain1=Xtrain[index,:]
        Ytrain1 = Y_train_X20[index]

        index = Y_train_X20 == 2
        Xtrain2=Xtrain[index,:]
        Ytrain2 = Y_train_X20[index]

        index = Y_train_X20 == 3
        Xtrain3=Xtrain[index,:]
        Ytrain3 = Y_train_X20[index]


        index = Y_test_X20 == 1
        Xtest1=XtestF[index,:]
        Ytest1 = Y_test_X20[index]
        index = Y_test_X20 == 2
        Xtest2 =XtestF[index,:]
        Ytest2 = Y_test_X20[index]
        index = Y_test_X20 == 3
        Xtest3=XtestF[index,:]
        Ytest3 = Y_test_X20[index]

        x = Xtrain
        y = Y_train_X20

        Train_X1 = Xtrain1
        Train_X2 = Xtrain2
        Train_X3 = Xtrain3
        Y_Ktrain1 = Ytrain1
        Y_Ktrain2 = Ytrain2
        Y_Ktrain3 = Ytrain3
        print('Train',Train_X1.shape,Train_X2.shape,Train_X3.shape)
        Test_X1 = Xtest1
        Test_X2 = Xtest2
        Test_X3 = Xtest3
        Y_Ktest1 = Ytest1
        Y_Ktest2 = Ytest2
        Y_Ktest3 = Ytest3
        print('Test:',Test_X1.shape, Test_X2.shape, Test_X3.shape)

        kmeans = Train_X2
        Ytrain2K = Y_Ktrain2



        Name_test = ["3VS.1&2","1VS.2&3"
            ,"1Vs2","1Vs3","2Vs3","2VS.1&3","1Vs2AllT","1Vs3AllT","2Vs3AllT"]




        ScoreTaskByKval = []
        ClassifierTask = []
        Prueba = []

        for z in range(len(Name_test)):
            print(Name_test[z])
            #3vs1&2
            if z==0:
                Xtotal = np.concatenate((Train_X1, kmeans, Train_X3), axis=0)
                Ytotal = np.concatenate((Y_Ktrain1*0, Ytrain2K * 0, Y_Ktrain3/3), axis=0)
                Xtesttotal = np.concatenate((Test_X1, Test_X2, Test_X3), axis=0)
                Ytesttotal = np.concatenate((Y_Ktest1*0, Y_Ktest2 * 0, Y_Ktest3/3 ), axis=0)

            #1vs2&3
            elif z == 1:
                Xtotal = np.concatenate((Train_X1, kmeans, Train_X3), axis=0)
                Ytotal = np.concatenate((Y_Ktrain1 , Ytrain2K * 0, Y_Ktrain3*0), axis=0)
                Xtesttotal = np.concatenate((Test_X1, Test_X2, Test_X3), axis=0)
                Ytesttotal = np.concatenate((Y_Ktest1 , Y_Ktest2 * 0, Y_Ktest3 *0), axis=0)
             #1vs2
            elif z==2:
                Xtotal = np.concatenate((Train_X1, kmeans), axis=0)
                Ytotal = np.concatenate((Y_Ktrain1 , Ytrain2K * 0), axis=0)
                Xtesttotal = np.concatenate((Test_X1, Test_X2), axis=0)
                Ytesttotal = np.concatenate((Y_Ktest1 , Y_Ktest2 * 0), axis=0)

            #1vs3
            elif z==3:
                Xtotal = np.concatenate((Train_X1, Train_X3), axis=0)
                Ytotal = np.concatenate((Y_Ktrain1 , Y_Ktrain3 *0), axis=0)
                Xtesttotal = np.concatenate((Test_X1, Test_X3), axis=0)
                Ytesttotal = np.concatenate((Y_Ktest1, Y_Ktest3 *0), axis=0)


            #2vs3
            elif z == 4:
                Xtotal = np.concatenate(( kmeans, Train_X3), axis=0)
                Ytotal = np.concatenate(( Ytrain2K/2 , Y_Ktrain3 *0), axis=0)
                Xtesttotal = np.concatenate(( Test_X2, Test_X3), axis=0)
                Ytesttotal = np.concatenate(( Y_Ktest2/2, Y_Ktest3*0), axis=0)

           #2vs1&3
            elif z == 5:

                Xtotal = np.concatenate((Train_X1, kmeans, Train_X3), axis=0)
                Ytotal = np.concatenate((Y_Ktrain1 * 0, Ytrain2K/2, Y_Ktrain3*0), axis=0)
                Xtesttotal = np.concatenate((Test_X1, Test_X2, Test_X3), axis=0)
                Ytesttotal = np.concatenate((Y_Ktest1 * 0, Y_Ktest2/2, Y_Ktest3*0), axis=0)
            ## PARA LAS PRUEBAS DE VALIDACION
            #1 VS 3
            if z==6:
                Xtotal = np.concatenate((Train_X1, Train_X3), axis=0)
                Ytotal = np.concatenate((Y_Ktrain1 , Y_Ktrain3), axis=0)
                Xtesttotal = np.concatenate((Test_X1, Test_X2, Test_X3), axis=0)
                Ytesttotal = np.concatenate((Y_Ktest1, Y_Ktest2 , Y_Ktest3 ), axis=0)

            #1 vS 2
            if z == 7:
                Xtotal = np.concatenate((Train_X1, kmeans), axis=0)
                Ytotal = np.concatenate((Y_Ktrain1 , Ytrain2K ), axis=0)
                Xtesttotal = np.concatenate((Test_X1, Test_X2, Test_X3), axis=0)
                Ytesttotal = np.concatenate((Y_Ktest1 , Y_Ktest2 , Y_Ktest3 ), axis=0)

            # 2 vS 3
            if z == 8:
                Xtotal = np.concatenate(( kmeans, Train_X3), axis=0)
                Ytotal = np.concatenate((Ytrain2K , Y_Ktrain3 ), axis=0)

                Xtesttotal = np.concatenate((Test_X1, Test_X2, Test_X3), axis=0)
                Ytesttotal = np.concatenate((Y_Ktest1 , Y_Ktest2 , Y_Ktest3 ), axis=0)

            X_train = Xtotal
            X_test = Xtesttotal
            y_train = Ytotal
            y_test = Ytesttotal

            y_train = Ytotal
            y_test = Ytesttotal


            # iterate over classifiers
            score_ClassifierForTask = []

            for name, clf in zip(names, classifiers):
                Prueba.append(Name_test[z])
                ClassifierTask.append(name)
                print('Task',name)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                y_pred_scores = clf.predict_proba(X_test)
#                confusion = confusion_matrix(y_test,y_pred)
                fscore_aux = fbeta_score(y_test, y_pred, average='macro', beta=0.5)
                print(fscore_aux)
                if np.int(np.unique(y_test).shape[0])<=2:
                    RocTrain = roc_auc_score(y_test, y_pred_scores[:,1])
                #RocTrain = roc_auc_score(y_test, y_pred_scores)
                else:
                    RocTrain = 0

                score_ClassifierForTask.append(fscore_aux)
            ScoreTaskByKval.extend(score_ClassifierForTask)
        ScoreAllKvalues.append(ScoreTaskByKval)

    objectExp = {'ScoreAllKvalues':ScoreAllKvalues, 'Experimento': Experimento, 'KfoldP':KfoldP, 'Prueba':Prueba[0],'ClassifierTask':ClassifierTask}
    with open(FileResults, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(objectExp, f)

with open(FileResults,'rb') as f:
        loaded_obj = pickle.load(f)

ScoreAllKvalues = loaded_obj['ScoreAllKvalues']
Experimento = loaded_obj['Experimento']
KfoldP = loaded_obj['KfoldP']
Prueba = loaded_obj['Prueba']
ClassifierTask = loaded_obj['ClassifierTask']






Metrica = np.array(ScoreAllKvalues)
df = pd.DataFrame(Metrica, index = [Experimento,KfoldP], columns = [Prueba,ClassifierTask])

#Definir los valores mas altos por experimento
ValMaxExpe = []
NameMaxExpe = []
for ExpDF in df.columns:
    valMeanKfolds =df[ExpDF].mean(level=0)
    NameMaxExpe.append(valMeanKfolds.idxmax())
    ValMaxExpe.append(valMeanKfolds.max())
DF_best = pd.Series(ValMaxExpe,index=[Prueba,NameMaxExpe])


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

## One vs all evaluation
Sel_bestOneVsAll = [24,76, 4]
oneVsAll = np.array([])
oneVsAllTest = np.array([])

oneVsOne = np.array([])

for best in Sel_bestOneVsAll:
    #search file max value
    idxF = np.where(all_train_val[:,best]==np.float(lista_train[2,best]))
    idxF = np.int(idxF[0][0])
    print(idxF)


    print(lista_train[0,best])
    print(lista_train[1, best])
    print('Fscore:',lista_train[2, best])

    #take the same column of the experiment and concatenate all folds
    print(files[idxF])
    valuesT = ScoresLearners_filesTrain[idxF]
    valuesTest = ScoresLearners_filesTest[idxF]
    #To define which is two higher probs and then evaluated with a classifier
    mC = concatenateKFold(valuesT,best)
    mCTest = concatenateKFold(valuesTest, best)

    if oneVsAll.size==0:
        oneVsAll = mC
        oneVsAllTest = mCTest
        allLabels = concatenateKFold(LabelsLearners_filesTrain[idxF],best)
    else:
        oneVsAll = np.vstack((oneVsAll,mC))
        oneVsAllTest = np.vstack((oneVsAllTest, mCTest))


oneVsAllOrig = oneVsAll.copy()
idxHG = np.argmin(oneVsAll,axis=0)

#onevsone

#
#
oneVsOne = np.array([])

idxF = np.where(all_train_val[:,95]==np.float(lista_train[2,95]))
idxF=np.int(idxF[0][0])
valuesT = ScoresLearners_filesTest[idxF]
OneVsTwo = concatenateKFold(valuesT,95)

#oneVsThree
idxF = np.where(all_train_val[:,110]==np.float(lista_train[2,110]))
idxF=np.int(idxF[0][0])
valuesT = ScoresLearners_filesTrain[idxF]
OneVsThree = concatenateKFold(valuesT,110)
#
 #TwoVsThree
idxF = np.where(all_train_val[:,125]==np.float(lista_train[2,125]))
idxF=np.int(idxF[0][0])
valuesT = ScoresLearners_filesTrain[idxF]
TwoVsThree = concatenateKFold(valuesT,125)

claseTotal = np.ones(len(idxHG))*100
for nt in range(len(idxHG)):
     if idxHG[nt] == 0:
         #2vs3
         cvalue = TwoVsThree[nt]
         if cvalue > 0.5:
             SelC = 2
         else:
             SelC = 3
         claseTotal[nt] = SelC
     elif idxHG[nt] == 1:
         #1vs3
         cvalue = OneVsThree[nt]
         if cvalue > 0.5:
             SelC = 1
         else:
             SelC = 3
         claseTotal[nt] = SelC
     elif idxHG[nt] ==2 :
#         #1vs2
         cvalue = OneVsTwo[nt]
         if cvalue > 0.5:
             SelC = 1
         else:
             SelC = 2
         claseTotal[nt] = SelC

allLabels = concatenateKFold(LabelsLearners_filesTrain[4],110)
claseTotal = np.argmax(oneVsAll,axis=0)+1
print(confusion_matrix(allLabels,claseTotal))
print(f1_score(allLabels, claseTotal , average='macro'))
#allLabelsTest = concatenateKFold(LabelsLearners_filesTest[6],110)
allLabelsTest = LabelsLearners_filesTest[6][4][110]
claseTotalTest = np.argmax(oneVsAllTest,axis=0)+1
print(confusion_matrix(allLabelsTest,claseTotalTest[0:124]))
print(f1_score(allLabelsTest , claseTotalTest[0:124] , average='macro'))
print('Ya ?')
















