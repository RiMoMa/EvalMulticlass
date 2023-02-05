def ParallelMatClassify(fileTrain,Experimento,KfoldP,RutaMAT,names,classifiers):
    import pandas as pd
    import numpy as np
    import glob
    import hdf5storage
    from sklearn.metrics import fbeta_score
    from sklearn import preprocessing

    print(fileTrain)
    print(len(fileTrain))
    fileTR = fileTrain
    fileTE = fileTrain[0:fileTR.find('TRAIN')] + 'TEST' + fileTrain[fileTR.find('TRAIN') + 5::]

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
        fileMatTr = sorted(glob.glob(RutaMAT + TrainDirDF.iloc[TrFile].ids + '*.mat'))
        print(fileMatTr[0])
        # if not(os.path.exists(fileMatTr[0]):
        file_mat = hdf5storage.loadmat(fileMatTr[0])
        Xtrain = file_mat['celda_matriz']
        Xtrain = Xtrain[0][0]
        Ytrain = Xtrain[:, -1]
        Xtrain = Xtrain[:, :-1]
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

    Xtrain = Xtrain[:, :].astype(np.float32)
    XtestF = Xtest[:, :].astype(np.float32)

    min_max_scaler = preprocessing.MinMaxScaler().fit(Xtrain[:, 0:816])
    Xtrain[:, 816::] = min_max_scaler.transform(Xtrain[:, 0:816])
    XtestF[:, 816::] = min_max_scaler.transform(XtestF[:, 816])
    # min_max_scaler = preprocessing.MinMaxScaler().fit(Xtrain[:,816::])
    # Xtrain[:,816::] = min_max_scaler.transform(Xtrain[:,816::])
    # XtestF[:,816::] = min_max_scaler.transform(XtestF[:,816::])

    Y_train_X20 = Ytrain
    Y_test_X20 = Ytest

    """Find arrays"""
    index = Y_train_X20 == 0
    Xtrain1 = Xtrain[index, :]
    Ytrain1 = Y_train_X20[index]

    index = Y_train_X20 == 1
    Xtrain2 = Xtrain[index, :]
    Ytrain2 = Y_train_X20[index]

    index = Y_test_X20 == 0
    Xtest1 = XtestF[index, :]
    Ytest1 = Y_test_X20[index]
    index = Y_test_X20 == 1
    Xtest2 = XtestF[index, :]
    Ytest2 = Y_test_X20[index]

    x = Xtrain
    y = Y_train_X20

    Train_X1 = Xtrain1
    Train_X2 = Xtrain2
    Y_Ktrain1 = Ytrain1
    Y_Ktrain2 = Ytrain2
    print('Train', Train_X1.shape, Train_X2.shape)
    Test_X1 = Xtest1
    Test_X2 = Xtest2
    Y_Ktest1 = Ytest1
    Y_Ktest2 = Ytest2
    print('Test:', Test_X1.shape, Test_X2.shape)

    kmeans = Train_X2
    Ytrain2K = Y_Ktrain2

    Name_test = ["1Vs2"]  # ,"1Vs2AllT","1Vs3AllT","2Vs3AllT"]

    ScoreTaskByKval = []
    ClassifierTask = []
    Prueba = []

    for z in range(len(Name_test)):
        print(Name_test[z])

        # 1vs2
        if z == 0:
            Xtotal = np.concatenate((Train_X1, kmeans), axis=0)
            Ytotal = np.concatenate((Y_Ktrain1, Ytrain2K), axis=0)
            Xtesttotal = np.concatenate((Test_X1, Test_X2), axis=0)
            Ytesttotal = np.concatenate((Y_Ktest1, Y_Ktest2), axis=0)

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
        results = []
        print('orden de :' + Experimento[0] + KfoldP[0])

        for name, clf in zip(names, classifiers):
            print('Task', name)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            fscore_aux = fbeta_score(y_test, y_pred, average='binary', beta=1)
            print(fscore_aux)
            score_ClassifierForTask.append(fscore_aux)

        ScoreTaskByKval.extend(score_ClassifierForTask)

    return ScoreTaskByKval


