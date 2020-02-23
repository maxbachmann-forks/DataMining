import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn import preprocessing
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_validate
from scipy import sparse
import xgboost as xgb



if __name__ == "__main__":

    train_data = pd.read_csv('./features', sep=',')
    test_data = pd.read_csv('./features_test', sep=',')

    train_data['Combined'] = train_data['Question1'] + ' ' + train_data['Question2']

    vectorizer = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS)
    vectorizer.fit(train_data['Combined'].values.astype('str'))

    #transform each question in the train set
    trainq1 = vectorizer.transform(train_data['Question1'].values.astype('str'))
    trainq2 = vectorizer.transform(train_data['Question2'].values.astype('str'))

    # transform each question in the test set
    testq1 = vectorizer.transform(test_data['Question1'].values.astype('str'))
    testq2 = vectorizer.transform(test_data['Question2'].values.astype('str'))

    #create the added features
    Jmatrix = sparse.csr_matrix((train_data['Jsimilarity']).to_numpy().reshape(-1, 1))
    Fmatrix = sparse.csr_matrix((train_data['Token_set_ratio']).to_numpy().reshape(-1, 1))

    TestJmatrix = sparse.csr_matrix((test_data['Jsimilarity']).to_numpy().reshape(-1, 1))
    TestFmatrix = sparse.csr_matrix((test_data['Token_set_ratio']).to_numpy().reshape(-1, 1))

    train_set = hstack([trainq1, trainq2])
    m = hstack([Fmatrix, Jmatrix])
    normalizer = preprocessing.Normalizer()
    m = normalizer.fit_transform(m)
    train_set = hstack([train_set,m])


    # SVM classifier
    clf = LinearSVC(random_state=0, tol=1e-5, C=0.6, loss='hinge', max_iter=100000)

    #xgb classifier just for kaggle

    clf= xgb.XGBClassifier(random_state=1, learning_rate=0.01)

    print('training')
    clf.fit(train_set,train_data['IsDuplicate'])

    test_set = hstack([testq1, testq2])
    m = hstack([TestFmatrix, TestJmatrix])
    normalizer = preprocessing.Normalizer()
    m = normalizer.fit_transform(m)
    test_set = hstack([test_set, m])

    print test_set.shape

    print('predicting')
    predictions = clf.predict(test_set)


    prediction = pd.DataFrame(data={"Predicted": predictions}, index=test_data['Id'])
    prediction.to_csv('testSet_categories.csv')