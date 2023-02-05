
print('Put the folder path:')
# input
FolderResults = '/mnt/md0/Histopathology/Resultados_Prueba/resultados_017_Deep2/'
#str(input('Put the folder path:'))
print('Path added:'+FolderResults)
NameResults = str(input('Put the results Name:'))
FileResults='/mnt/md0/Histopathology/' + NameResults +'.pkl'


RutaMAT = '/mnt/md0/Histopathology/Resultados_Prueba/resultados_021/'

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


# Parallelizing using Pool.apply()

import multiprocessing as mp
from ParallelClassificationASYNC import ParallelClassificationASYNCFunc
from ParallelClassificationASYNC import collect_result

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
from openKfold import openKfold
h = .02  # step size in the mesh
import pickle
from sklearn.base import clone
from sklearn.preprocessing import Binarizer
#pathTrainArray = 'XtrainFeatures2019.h5'
#pathTestArray = 'XtestFeatures2019.h5'
import hdf5storage

"""PARTE DE TEST"""

names = [
    "Nearest Neighbors",
#    "Linear SVM",
  #  "RBF SVM",
    #"Chi2 SVM",
   # "Cosine SVM",
 #   "GHK SVM",
 #   "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "AdaBoost",
    "Naive Bayes",
    "LogisticRegression",
    "GradientBoosting",
    "RUSBOOST"
    ]

classifiers = [
    KNeighborsClassifier(6),
 #   SVC(kernel="linear", C=0.1,probability=True),
  #  SVC(gamma=2, C=1,probability=True),
    #SVC(kernel=Chi2(), C=1,probability=True),
  #  SVC(kernel=Cossim(), C=1,probability=True),
 #   SVC(kernel=GeneralizedHistogramIntersection(), C=1,probability=True),
   # GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=50),
    RandomForestClassifier(max_depth=1, n_estimators=50, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(priors=None, var_smoothing=1e-04),
    LogisticRegression(solver='sag',max_iter=200),
    GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=50, random_state=0),
    RUSBoostClassifier(random_state=0)

]

#h5f = h5py.File(pathTestArray, 'r')
#XtestF = h5f['XtestAuto'][:]
#Ytest = h5f['Y_test'][:]
#h5f.close()
#print('Vector TEST LOAD!')


#files = glob.glob('/home/ricardo/Documentos/experimentosMay/')
#files=sorted(glob.glob('/home/ricardo/Documentos/SYNC/ExperimentosGrafoHaralickS/train*.mat'), key=os.path.basename)
files=sorted(glob.glob(FolderResults+'*TRAIN*.xls'), key=os.path.basename)
#/home/moncayor/moncayoR/ExperimentosGrafoHaralickS

import scipy.io as sio
from sklearn.model_selection import cross_val_score
Kfold = 10


Experimento = []
KfoldP = []

Metrica = []

if not(os.path.exists(FileResults)):
    ScoreAllKvalues=[]

    #files = files[0:15]


    for fileTrain in files:

        print(fileTrain)
        print(len(fileTrain))
        fileTR = fileTrain
        fileTE = fileTrain[0:fileTR.find('TRAIN')] + 'TEST' + fileTrain[fileTR.find('TRAIN')+5::]


        Experimento.append(fileTrain[fileTrain.find('Iteracion'):fileTrain.find('_kfold')])
        KfoldP.append(fileTrain[fileTrain.find('_kfold'):fileTrain.find('_TRAIN.xls')])
        print(fileTE)

        TrainDirDF = pd.read_excel(fileTR)
        TrainDirDF = TrainDirDF.astype(str)
        df2 = pd.DataFrame([TrainDirDF.columns[0]], columns=list(TrainDirDF.columns))
        TrainDirDF = TrainDirDF.append(df2, ignore_index=True)
        TrainDirDF.columns = ['ids']
        TestDirDF = pd.read_excel(fileTE)
        TestDirDF = TestDirDF.astype(str)
        df2 = pd.DataFrame([TestDirDF.columns[0]], columns=list(TestDirDF.columns))
        TestDirDF = TestDirDF.append(df2, ignore_index=True)
        TestDirDF.columns = ['ids']

        # vamos abrir todos los mat
        XtrainAll = []
        YtrainAll = []
        
        for TrFile in range(len(TrainDirDF)):
            fileMatTr = sorted(glob.glob(RutaMAT+TrainDirDF.iloc[TrFile].ids+'*.mat'))
            print(fileMatTr[0])
            #if not(os.path.exists(fileMatTr[0]):
            file_mat=hdf5storage.loadmat(fileMatTr[0])
            Xtrain = file_mat['celda_matriz']
            Xtrain = Xtrain[0][0]
            Ytrain = Xtrain[:,-1]
            Xtrain = Xtrain[:,:-1]
            XtrainAll.extend(Xtrain)
            YtrainAll.extend(Ytrain)

        Xtrain = np.array(XtrainAll)
        Ytrain = np.array(YtrainAll)

        XtestAll = []
        YtestAll = []
        for TrFile in range(len(TestDirDF)):
            fileMatTr = sorted(glob.glob(RutaMAT + TestDirDF.iloc[TrFile].ids + '*.mat'))
            file_mat = hdf5storage.loadmat(fileMatTr[0])
            Xtest = file_mat['celda_matriz']
            Xtest = Xtest[0][0]
            Ytest = Xtest[:, -1]
            Xtest = Xtest[:, :-1]
            XtestAll.extend(Xtest)
            YtestAll.extend(Ytest)
        Xtest = np.array(XtestAll)
        Ytest = np.array(YtestAll)


       # Ytrain = np.reshape(Ytrain,len(Ytrain))
      #  Ytest = np.reshape(Ytest,len(Ytest))

        Xtrain = Xtrain[:,:].astype(np.float32)
        XtestF = Xtest[:,:].astype(np.float32)

        ### aplying TF IDF

   #     IDF = np.log10(len(Xtrain) / (1 + np.sum(Xtrain>0, axis=0)))
    #    Xtrain = Xtrain * IDF
     #   XtestF = XtestF * IDF



        min_max_scaler = preprocessing.MinMaxScaler().fit(Xtrain[:,760::])
        Xtrain[:,760::] = min_max_scaler.transform(Xtrain[:,760::])
        XtestF[:,760::] = min_max_scaler.transform(XtestF[:,760::])
       # min_max_scaler = preprocessing.MinMaxScaler().fit(Xtrain[:,816::])
       # Xtrain[:,816::] = min_max_scaler.transform(Xtrain[:,816::])
       # XtestF[:,816::] = min_max_scaler.transform(XtestF[:,816::])


        Y_train_X20 = Ytrain
        Y_test_X20 = Ytest



        """Find arrays"""
        index = Y_train_X20 == 0
        Xtrain1=Xtrain[index,:]
        Ytrain1 = Y_train_X20[index]

        index = Y_train_X20 == 1
        Xtrain2=Xtrain[index,:]
        Ytrain2 = Y_train_X20[index]


        index = Y_test_X20 == 0
        Xtest1=XtestF[index,:]
        Ytest1 = Y_test_X20[index]
        index = Y_test_X20 == 1
        Xtest2 =XtestF[index,:]
        Ytest2 = Y_test_X20[index]


        x = Xtrain
        y = Y_train_X20

        Train_X1 = Xtrain1
        Train_X2 = Xtrain2
        Y_Ktrain1 = Ytrain1
        Y_Ktrain2 = Ytrain2
        print('Train',Train_X1.shape,Train_X2.shape)
        Test_X1 = Xtest1
        Test_X2 = Xtest2
        Y_Ktest1 = Ytest1
        Y_Ktest2 = Ytest2
        print('Test:',Test_X1.shape, Test_X2.shape)

        kmeans = Train_X2
        Ytrain2K = Y_Ktrain2



        Name_test = ["1Vs2"]#,"1Vs2AllT","1Vs3AllT","2Vs3AllT"]




        ScoreTaskByKval = []
        ClassifierTask = []
        Prueba = []

        for z in range(len(Name_test)):
            print(Name_test[z])

             #1vs2
            if z==0:
                Xtotal = np.concatenate((Train_X1, kmeans), axis=0)
                Ytotal = np.concatenate((Y_Ktrain1 , Ytrain2K ), axis=0)
                Xtesttotal = np.concatenate((Test_X1, Test_X2), axis=0)
                Ytesttotal = np.concatenate((Y_Ktest1 , Y_Ktest2 ), axis=0)


            #     Ytesttotal = np.concatenate((Y_Ktest1 , Y_Ktest2 , Y_Ktest3 ), axis=0)

            X_train = Xtotal
            X_test = Xtesttotal
            y_train = Ytotal
            y_test = Ytesttotal

            y_train = Ytotal
            y_test = Ytesttotal


            # iterate over classifiers
            score_ClassifierForTask = []

            # Step 1: Init multiprocessing.Pool()
            classifiers_cloned = []
            results=[]
            pool = mp.Pool(10)

            for CL in range(len(classifiers)):
                classifiers_cloned.append(clone(classifiers[CL]))
            ## esta variable id_Classificador para ordenar los resultados asyncronos
            id_Classificador = np.arange(len(classifiers))


            # Step 2: `pool.apply` the `howmany_within_range()`
            score_ClassifierForTask = [pool.apply_async(ParallelClassificationASYNCFunc, args=(name, clf,id_one, X_train, y_train, X_test, y_test)) for name, clf,id_one in zip(names, classifiers_cloned,id_Classificador)]
            results = [r.get() for r in score_ClassifierForTask]
            #for name, clf, id_one in zip(names, classifiers_cloned, id_Classificador):
            #    print('paralelo o no??')
            #    pool.apply_async(ParallelClassificationASYNCFunc, args=(name, clf,id_one, X_train, y_train, X_test, y_test),callback=collect_result)

            # Step 3: Don't forget to close
            pool.close()
            pool.join()
              # postpones the execution of next line of code until all processes in the queue are done.

            # Step 5: Sort results [OPTIONAL]

            print('orden de :' + Experimento[0] + KfoldP[0])

            for name, clf,idPrt in zip(names, classifiers,id_Classificador):
                Prueba.append(Name_test[z])
                ClassifierTask.append(name)
                print('Task',name)
                print(results[idPrt])

            ScoreTaskByKval.extend(results)
        ScoreAllKvalues.append(ScoreTaskByKval)

    objectExp = {'ScoreAllKvalues':ScoreAllKvalues, 'Experimento': Experimento, 'KfoldP':KfoldP, 'Prueba':Prueba,'ClassifierTask':ClassifierTask}
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
DF_best = pd.Series(ValMaxExpe,index=[Prueba,NameMaxExpe,ClassifierTask])

## Cuales son los mejores entonces?

### parte 1 vs 2 and 3


indexes = []
data = []
for n in range(6):
    if n==0:
        indexes = DF_best.max(level=0).index[n]
        data = DF_best.loc[DF_best.max(level=0).index[n]].idxmax()
        score_max = DF_best.loc[DF_best.max(level=0).index[n]].max()

        TaskBest = pd.Series(score_max,index=[[indexes],[data[0]],[data[1]]])
    else:
        indexes = DF_best.max(level=0).index[n]
        data = DF_best.loc[DF_best.max(level=0).index[n]].idxmax()
        score_max = DF_best.loc[DF_best.max(level=0).index[n]].max()

        TaskBest2 = pd.Series(score_max, index=[[indexes],[data[0]],[data[1]]])
        TaskBest = TaskBest.append(TaskBest2)






TrainFileBest =FolderResults +'trainNorm_augmented_'+\
               TaskBest.index[1][1]
TestFileBest =FolderResults +'testNorm_augmented_'+\
               TaskBest.index[1][1]
clasifierABC = clone(classifiers[names.index(TaskBest.index[1][2])])

TrainFileBest2 =FolderResults +'trainNorm_augmented_'+\
               TaskBest.index[4][1]
TestFileBest2 =FolderResults +'testNorm_augmented_'+\
               TaskBest.index[4][1]
clasifierBC = clone(classifiers[names.index(TaskBest.index[4][2])])

ListFolds= ['_Kfold1.mat','_Kfold2.mat','_Kfold3.mat','_Kfold4.mat','_Kfold5.mat']

y_testAll=[]
Y_predAll=[]

for n in range(5):
    TrainFileK = TrainFileBest+ListFolds[n]
    TestFileK = TestFileBest + ListFolds[n]
    TrainFileK2 = TrainFileBest2+ListFolds[n]
    TestFileK2 = TestFileBest2 + ListFolds[n]
    Xtrain1,Ytrain1,Xtrain2,Ytrain2,Xtrain3,Ytrain3,XtestF,Ytest = openKfold(TrainFileK,TestFileK)
    Xtrain1b, Ytrain1b, Xtrain2b, Ytrain2b, Xtrain3b, Ytrain3b, XtestFb, Ytest = openKfold(TrainFileK2, TestFileK2)

    Mdl1Vs2and3 = clasifierABC
    #Mdl2vs3 = LogisticRegression(solver='liblinear')
    Mdl2vs3 = clasifierBC

  #  Mdl2vs3 = SVC(kernel=GeneralizedHistogramIntersection(), C=1, probability=True)

    Xtotal = np.concatenate((Xtrain1, Xtrain2, Xtrain3), axis=0)
    Ytotal = np.concatenate((Ytrain1 , Ytrain2* 0, Ytrain3*0), axis=0)
    Mdl1Vs2and3.fit(Xtotal,Ytotal)

    X2and3 = np.concatenate( (Xtrain2b, Xtrain3b), axis=0)
    Y2and3 = np.concatenate((Ytrain2b , Ytrain3b), axis=0)
    Mdl2vs3.fit(X2and3,Y2and3)

    #Ypred1Vsrest =Mdl1Vs2and3.predict(XtestF)
    Ypred1Vsrest = Mdl1Vs2and3.predict_proba(XtestF)
    Ypred1Vsrest = Ypred1Vsrest[:, 1] > 0.45
    Ypred1Vsrest = np.uint8(Ypred1Vsrest)

    XtestF2=XtestFb[Ypred1Vsrest==0,:]
    YpredictRest = Mdl2vs3.predict(XtestF2)
    Ypred1Vsrest[Ypred1Vsrest==0]=YpredictRest

    Y_predAll.extend(Ypred1Vsrest)
    y_testAll.extend(Ytest)

print('mejores:')
print(TaskBest)
print(confusion_matrix(y_testAll,Y_predAll))
print('Y ahora el orden...')


### parte 3 vs 1 and 2
y_testAll=[]
Y_predAll=[]

TrainFileBest =FolderResults + 'trainNorm_augmented_'+\
               TaskBest.index[0][1]
TestFileBest =FolderResults +'testNorm_augmented_'+\
               TaskBest.index[0][1]
clasifierABC = clone(classifiers[names.index(TaskBest.index[0][2])])

TrainFileBest2 =FolderResults + 'trainNorm_augmented_'+\
               TaskBest.index[2][1]
TestFileBest2 =FolderResults + 'testNorm_augmented_'+\
               TaskBest.index[2][1]
clasifierBC = clone(classifiers[names.index(TaskBest.index[2][2])])

for n in range(5):
    TrainFileK = TrainFileBest + ListFolds[n]
    TestFileK = TestFileBest + ListFolds[n]
    TrainFileK2 = TrainFileBest2 + ListFolds[n]
    TestFileK2 = TestFileBest2 + ListFolds[n]
    Xtrain1, Ytrain1, Xtrain2, Ytrain2, Xtrain3, Ytrain3, XtestF, Ytest = openKfold(TrainFileK, TestFileK)
    Xtrain1b, Ytrain1b, Xtrain2b, Ytrain2b, Xtrain3b, Ytrain3b, XtestFb, Ytest = openKfold(TrainFileK2, TestFileK2)

    Mdl3Vs1and2 = clasifierABC
   # Mdl2vs3 = LogisticRegression(solver='liblinear')
    Mdl1vs2 = clasifierBC
    Xtotal = np.concatenate((Xtrain1, Xtrain2, Xtrain3), axis=0)
    Ytotal = np.concatenate((Ytrain1*0 , Ytrain2*0, Ytrain3), axis=0)
    Mdl3Vs1and2.fit(Xtotal,Ytotal)

    X1and2 = np.concatenate( (Xtrain1b, Xtrain2b), axis=0)
    Y1and2 = np.concatenate((Ytrain1b , Ytrain2b), axis=0)
    Mdl1vs2.fit(X1and2,Y1and2)

    Ypred2Vsrest =Mdl3Vs1and2.predict(XtestF)
    XtestF2=XtestFb[Ypred2Vsrest==0,:]
    YpredictRest = Mdl1vs2.predict(XtestF2)
    Ypred2Vsrest[Ypred2Vsrest==0]=YpredictRest

    Y_predAll.extend(Ypred2Vsrest)
    y_testAll.extend(Ytest)


print(confusion_matrix(y_testAll,Y_predAll))
print('Y ahora el orden...')


### parte 2 vs 1 and 3
y_testAll=[]
Y_predAll=[]

TrainFileBest =FolderResults + 'trainNorm_augmented_'+\
               TaskBest.index[5][1]
TestFileBest =FolderResults +'testNorm_augmented_'+\
               TaskBest.index[5][1]
clasifierABC = clone(classifiers[names.index(TaskBest.index[5][2])])

TrainFileBest2 =FolderResults + 'trainNorm_augmented_'+\
               TaskBest.index[3][1]
TestFileBest2 =FolderResults + 'testNorm_augmented_'+\
               TaskBest.index[3][1]
clasifierBC = clone(classifiers[names.index(TaskBest.index[3][2])])

for n in range(5):
    TrainFileK = TrainFileBest + ListFolds[n]
    TestFileK = TestFileBest + ListFolds[n]
    TrainFileK2 = TrainFileBest2 + ListFolds[n]
    TestFileK2 = TestFileBest2 + ListFolds[n]
    Xtrain1, Ytrain1, Xtrain2, Ytrain2, Xtrain3, Ytrain3, XtestF, Ytest = openKfold(TrainFileK, TestFileK)
    Xtrain1b, Ytrain1b, Xtrain2b, Ytrain2b, Xtrain3b, Ytrain3b, XtestFb, Ytest = openKfold(TrainFileK2, TestFileK2)

    Mdl2Vs1and3 = clasifierABC
   # Mdl2vs3 = LogisticRegression(solver='liblinear')
    Mdl1vs3 = clasifierBC
    #Mdl1vs3 =
    Xtotal = np.concatenate((Xtrain1, Xtrain2, Xtrain3), axis=0)
    Ytotal = np.concatenate((Ytrain1*0 , Ytrain2, Ytrain3*0), axis=0)
    Mdl2Vs1and3.fit(Xtotal,Ytotal)

    X1and3 = np.concatenate( (Xtrain1b, Xtrain3b), axis=0)
    Y1and3 = np.concatenate((Ytrain1b , Ytrain3b), axis=0)
    Mdl1vs3.fit(X1and3,Y1and3)

    Ypred2Vsrest =Mdl2Vs1and3.predict(XtestF)
    XtestF2=XtestFb[Ypred2Vsrest==0,:]
    YpredictRest = Mdl1vs3.predict(XtestF2)
    Ypred2Vsrest[Ypred2Vsrest==0]=YpredictRest

    Y_predAll.extend(Ypred2Vsrest)
    y_testAll.extend(Ytest)


print(confusion_matrix(y_testAll,Y_predAll))