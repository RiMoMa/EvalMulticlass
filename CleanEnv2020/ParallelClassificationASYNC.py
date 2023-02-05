def ParallelClassificationASYNCFunc(name,clf,id_one,X_train,y_train,X_test,y_test):
    from sklearn.metrics import fbeta_score
    print('Task',name)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    fscore_aux = fbeta_score(y_test, y_pred, average='binary', beta=1)
    print(fscore_aux)
    return fscore_aux

def collect_result(result):
    global results
    results.append(result)