#FolderResults = '/mnt/md0/ricardo/ResultadosCurveletsSmallSetCosine/'
#FolderResults = '/mnt/md0/ricardo/ResultadosCurveletsSmallSetNoiselet/'
#FolderResults = '/mnt/md0/ricardo/ResultadosJune17CurveletsHaralicj/'
#FolderResults = '/mnt/md0/ricardo/ResultadosJune17CurveletsL/'
print('Put the folder path:')
# input
FolderResults = str(input('Put the folder path:'))

# output
print('Path added:'+FolderResults)
NameResults = str(input('Put the results Name:'))
FileResults=FolderResults + NameResults +'.pkl'
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
from ParallelClassification import ParallelClassification
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

#h5f = h5py.File(pathTrainArray, 'r')
#Xtrain = h5f['XtrainAuto'][:]
#Ytrain = h5f['Y_train_X20'][:]
#h5f.close()
#print('Vector TRAINING LOAD!')

"""PARTE DE TEST"""

names = [
    "Nearest Neighbors",
  #  "Linear SVM",
    #"RBF SVM",
    # "Chi2 SVM",
 #   "Cosine SVM",
   # "GHK SVM",
#    "Gaussian Process",
   # "Decision Tree",
#    "Random Forest",
#    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
 #   "QDA",
 #   "LogisticRegression",
 #   "GradientBoosting",
 #   "RUSBOOST"
    #"Voting1vs3", "Voting1v2", "NewVoting",
    #"onevsallVoting"
    ]

classifiers = [
    KNeighborsClassifier(3,weights='distance',leaf_size=10),
  #  SVC(kernel="linear", C=0.1,probability=True),
    #SVC(gamma=2, C=1,probability=True),
    #SVC(kernel=Chi2(), C=1,probability=True),
   # SVC(kernel=Cossim(), C=1,probability=True),
   # SVC(kernel=GeneralizedHistogramIntersection(), C=1,probability=True),
  #  GaussianProcessClassifier(1.0 * RBF(1.0)),
   # DecisionTreeClassifier(max_depth=10),
   # RandomForestClassifier(max_depth=1, n_estimators=12, max_features=1),
   # MLPClassifier(alpha=1),
    AdaBoostClassifier(n_estimators=2,learning_rate=1),
    GaussianNB(priors=None, var_smoothing=1e-11),
   # QuadraticDiscriminantAnalysis(),
   # LogisticRegression(solver='lbfgs'),
   # GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0),
   # RUSBoostClassifier(random_state=0),
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
files=sorted(glob.glob(FolderResults+'TRAIN*.mat'), key=os.path.basename)
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

        ### aplying TF IDF

   #     IDF = np.log10(len(Xtrain) / (1 + np.sum(Xtrain>0, axis=0)))
    #    Xtrain = Xtrain * IDF
     #   XtestF = XtestF * IDF



        ### Normalized according pyramidal structure
        Xtrain = Xtrain[:,0:299]
        XtestF = XtestF[:,0:299]
       # portionX = np.int(Xtrain.shape[1]/21*5)
       # Xtrain[:,portionX::] = Xtrain[:,portionX::]*2
       # XtestF[:, portionX::] = XtestF[:, portionX::] * 2
       # Xtrain[:,::] = Xtrain[:,:]*(1/4)
       # XtestF[:,::] = XtestF[:, :] *(1/4)


    #    min_max_scaler = preprocessing.MinMaxScaler().fit(Xtrain)
    #    Xtrain = min_max_scaler.transform(Xtrain)
    #    XtestF = min_max_scaler.transform(XtestF)




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
            ,"1Vs2","1Vs3","2Vs3","2VS.1&3"]#,"1Vs2AllT","1Vs3AllT","2Vs3AllT"]




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
                Ytotal = np.concatenate(( Ytrain2K*0 , Y_Ktrain3 /3), axis=0)
                Xtesttotal = np.concatenate(( Test_X2, Test_X3), axis=0)
                Ytesttotal = np.concatenate(( Y_Ktest2*0, Y_Ktest3/3), axis=0)

           #2vs1&3
            elif z == 5:

                Xtotal = np.concatenate((Train_X1, kmeans, Train_X3), axis=0)
                Ytotal = np.concatenate((Y_Ktrain1 , Ytrain2K*0, Y_Ktrain3/3), axis=0)
                Xtesttotal = np.concatenate((Test_X1, Test_X2, Test_X3), axis=0)
                Ytesttotal = np.concatenate((Y_Ktest1 , Y_Ktest2*0, Y_Ktest3/3), axis=0)
            ## PARA LAS PRUEBAS DE VALIDACION
            #1 VS 3
            # if z==6:
            #     Xtotal = np.concatenate((Train_X1, Train_X3), axis=0)
            #     Ytotal = np.concatenate((Y_Ktrain1 , Y_Ktrain3), axis=0)
            #     Xtesttotal = np.concatenate((Test_X1, Test_X2, Test_X3), axis=0)
            #     Ytesttotal = np.concatenate((Y_Ktest1, Y_Ktest2 , Y_Ktest3 ), axis=0)
            #
            # #1 vS 2
            # if z == 7:
            #     Xtotal = np.concatenate((Train_X1, kmeans), axis=0)
            #     Ytotal = np.concatenate((Y_Ktrain1 , Ytrain2K ), axis=0)
            #     Xtesttotal = np.concatenate((Test_X1, Test_X2, Test_X3), axis=0)
            #     Ytesttotal = np.concatenate((Y_Ktest1 , Y_Ktest2 , Y_Ktest3 ), axis=0)
            #
            # # 2 vS 3
            # if z == 8:
            #     Xtotal = np.concatenate(( kmeans, Train_X3), axis=0)
            #     Ytotal = np.concatenate((Ytrain2K , Y_Ktrain3 ), axis=0)
            #
            #     Xtesttotal = np.concatenate((Test_X1, Test_X2, Test_X3), axis=0)
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
            pool = mp.Pool(5)
            classifiers_cloned = []
            for CL in range(len(classifiers)):
                classifiers_cloned.append(clone(classifiers[CL]))


            # Step 2: `pool.apply` the `howmany_within_range()`
            score_ClassifierForTask = [pool.apply(ParallelClassification, args=(name, clf, X_train, y_train, X_test, y_test)) for name, clf in zip(names, classifiers_cloned)]

            # Step 3: Don't forget to close
            pool.close()

            for name, clf in zip(names, classifiers):
                Prueba.append(Name_test[z])
                ClassifierTask.append(name)
                print('Task',name)

            ScoreTaskByKval.extend(score_ClassifierForTask)
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






TrainFileBest =FolderResults +'TRAIN_'+\
               TaskBest.index[1][1]
TestFileBest =FolderResults +'TEST_'+\
               TaskBest.index[1][1]
clasifierABC = clone(classifiers[names.index(TaskBest.index[1][2])])

TrainFileBest2 =FolderResults +'TRAIN_'+\
               TaskBest.index[4][1]
TestFileBest2 =FolderResults +'TEST_'+\
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

TrainFileBest =FolderResults + 'TRAIN_'+\
               TaskBest.index[0][1]
TestFileBest =FolderResults +'TEST_'+\
               TaskBest.index[0][1]
clasifierABC = clone(classifiers[names.index(TaskBest.index[0][2])])

TrainFileBest2 =FolderResults + 'TRAIN_'+\
               TaskBest.index[2][1]
TestFileBest2 =FolderResults + 'TEST_'+\
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

TrainFileBest =FolderResults + 'TRAIN_'+\
               TaskBest.index[5][1]
TestFileBest =FolderResults +'TEST_'+\
               TaskBest.index[5][1]
clasifierABC = clone(classifiers[names.index(TaskBest.index[5][2])])

TrainFileBest2 =FolderResults + 'TRAIN_'+\
               TaskBest.index[3][1]
TestFileBest2 =FolderResults + 'TEST_'+\
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