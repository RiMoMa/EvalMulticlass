
def ParallelOpenMatFiles(TrainDirDF,RutaMAT):
    import glob
    import hdf5storage
    fileMatTr = sorted(glob.glob(RutaMAT + TrainDirDF +'*.mat'))
    print(fileMatTr[0])
    # if not(os.path.exists(fileMatTr[0]):
    file_mat = hdf5storage.loadmat(fileMatTr[0])
    Xtrain = file_mat['celda_matriz']
    Xtrain = Xtrain[0][0]
    Ytrain = Xtrain[:, -1]
    Xtrain = Xtrain[:, :-1]
    return (Xtrain,Ytrain)

     #   XtrainAll.extend(Xtrain)
     #   YtrainAll.extend(Ytrain)