def openKfold(TrainFileK,TestFileK):
    import numpy as np
    import scipy.io as sio
    from sklearn import preprocessing

    file_mat = sio.loadmat(TrainFileK)
    Xtrain = file_mat['X']
    Ytrain = file_mat['Y']

    file_mat = sio.loadmat(TestFileK)
    XtestF = file_mat['Xtest']
    Ytest = file_mat['Ytest']
    Ytrain = np.reshape(Ytrain, len(Ytrain))
    Ytest = np.reshape(Ytest, len(Ytest))
    ### Normalized according pyramidal structure
    Xtrain = Xtrain[:,0:299]
    XtestF = XtestF[:,0:299]
#    portionX = np.int(Xtrain.shape[1] / 21 * 5)
#    Xtrain[:, portionX::] = Xtrain[:, portionX::] * 2
#    XtestF[:, portionX::] = XtestF[:, portionX::] * 2
#    Xtrain[:, ::] = Xtrain[:, :] * (1 / 4)
#    XtestF[:, ::] = XtestF[:, :] * (1 / 4)

   # min_max_scaler = preprocessing.MinMaxScaler()
   # Xtrain = min_max_scaler.fit_transform(Xtrain)
   # XtestF = min_max_scaler.transform(XtestF)
    """Find arrays"""
    index = Ytrain == 1
    Xtrain1 = Xtrain[index, :]
    Ytrain1 = Ytrain[index]

    index = Ytrain == 2
    Xtrain2 = Xtrain[index, :]
    Ytrain2 = Ytrain[index]

    index = Ytrain == 3
    Xtrain3 = Xtrain[index, :]
    Ytrain3 = Ytrain[index]

    return Xtrain1,Ytrain1,Xtrain2,Ytrain2,Xtrain3,Ytrain3,XtestF,Ytest