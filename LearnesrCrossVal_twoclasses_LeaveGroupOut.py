
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#os.environ["CUDA_VISIBLE_DEVICES"]="2"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"

import sys
sys.path.append('/home/ricardo/PycharmProjects/pykernels-master/pykernels/')
sys.path.append('/home/ricardo/PycharmProjects/pykernels-master/pykernels/graph/')


import numpy as np
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
from pykernels.regular import Chi2,Cossim,GeneralizedHistogramIntersection
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score
h = .02  # step size in the mesh


"""PARTE DE TEST"""

names = [#"Nearest Neighbors",
        # "Linear SVM",
        # "RBF SVM",
        # "Chi2 SVM",
        # "Cosine SVM",
        # "GHK SVM",
       #  "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA", "LogisticRegression", "GradientBoosting",
         #"Voting1vs3",
       #  "Voting1v2", "NewVoting",
       #  "onevsallVoting"
         ]

classifiers = [
    #KNeighborsClassifier(3,n_jobs=4),
  #  linear_model.SGDClassifier(max_iter=1000, tol=1e-3,n_jobs=4),
  #  SVC(kernel="linear", C=0.025,probability=True),
  #  SVC(gamma=2, C=1,probability=True),
  #  SVC(kernel=Chi2(), C=1,probability=True),
  #  SVC(kernel=Cossim(), C=1,probability=True),
  #  SVC(kernel=GeneralizedHistogramIntersection(), C=1,probability=True),
   # GaussianProcessClassifier(1.0 * RBF(1.0),n_jobs=4),
    DecisionTreeClassifier(max_depth=10),
    RandomForestClassifier(max_depth=1, n_estimators=12, max_features=1,n_jobs=4),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(priors=None, var_smoothing=1e-04),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression(solver='lbfgs',n_jobs=4),
    GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
]




#files = glob.glob('/home/ricardo/Documentos/experimentosMay/')
#file=sorted(glob.glob('/home/ricardo/Documents/Doctorado/SYNC/all_codes/NoiseletsNuclei/ResultadosTrai/ResultsTrain_B1_F_TCGA-18-5592-01Z-00-DX1.mat'), key=os.path.basename)
file='/home/ricardo/Documents/Doctorado/SYNC/all_codes/NoiseletsNuclei/ResultadosTrai/ResultsTrain_B1_F_TCGA-18-5592-01Z-00-DX1.mat'
import scipy.io as sio
from sklearn.model_selection import cross_val_score

file_mat=sio.loadmat(file)
Xtrain = file_mat['X']
Ytrain = file_mat['Y']
Xtrain = Xtrain.astype(np.float32)
groups = Xtrain[:,4]
Xtrain = Xtrain[:,0:4]
#search and eliminate Nan rows
# print('Searching by Nan Values')
# RowsNan = []
# for n in range(0, Xtrain.shape[0]):
#     if np.isinf(Xtrain[n, :]).any() == True:
#         RowsNan.append(n)
#         print('value:', n, np.isinf(Xtrain[n, :]).any() == True)
# Xtrain = np.delete(Xtrain,RowsNan,0)
# Ytrain = np.delete(Ytrain,RowsNan,0)
ytrain = Ytrain[:,0]

logo = LeaveOneGroupOut()
logo.get_n_splits(Xtrain, ytrain, groups)

logo.get_n_splits(groups=groups)  # 'groups' is always required

print(logo)

RocKG_all=[]
FscoreKG_all = []
AccKG_all = []
for name, clf in zip(names, classifiers):

    print(clf)
    RocKG=[]
    FscoreKG = []
    AccKG = []
    counter = 1

    for train_index, test_index in logo.split(Xtrain, ytrain, groups):
       print('Evaluating Group:', counter)
       counter = counter + 1
       X_train, X_test = Xtrain[train_index], Xtrain[test_index]
       y_train, y_test = ytrain[train_index], ytrain[test_index]
       clf.fit(X_train, y_train)
       y_scores = clf.predict_proba(X_test)
       Y_pred = clf.predict(X_test)
       RocKG.append(roc_auc_score(y_test, y_scores[:,1]))
       FscoreKG.append(f1_score(y_test, Y_pred, average='macro'))
       AccKG.append(accuracy_score(y_test, Y_pred, normalize=True))
    RocKG = np.mean(RocKG)
    FscoreKG = np.mean (FscoreKG)
    AccKG = np.mean(AccKG)
    print(names)
    print('auc:',RocKG)
    print('fscore:',FscoreKG)
    print('Acc:',AccKG)

    RocKG_all.append(RocKG)
    FscoreKG_all.append(FscoreKG)
    AccKG_all.append(AccKG)


indice = FscoreKG_all.index(max(FscoreKG_all))
print('Best Classifier:\n',names[indice])
print('FINAL RESULT: ')
ACCF = AccKG_all[indice]
print('ACC:',AccKG_all[indice])
print('F-score Leave one out:',FscoreKG_all[indice])
print('AUC Leave one Out:',RocKG_all[indice])

indice = RocKG_all.index(max(RocKG_all))
print('Best Classifier:\n',names[indice])
print('FINAL RESULT: ')
ACCF = AccKG_all[indice]
print('ACC:',AccKG_all[indice])
print('F-score Leave one out:',FscoreKG_all[indice])
print('AUC Leave one Out:',RocKG_all[indice])











































