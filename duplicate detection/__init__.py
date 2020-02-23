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



if __name__ == "__main__":

    train_data = pd.read_csv('./features', sep=',')
    test_data = pd.read_csv('../dataset/datasets/q2b/test_without_labels.csv', sep=',')

    train_data['Combined'] = train_data['Question1'] + ' ' + train_data['Question2']

    vectorizer = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS)
    vectorizer.fit(train_data['Combined'].values.astype('str'))
    trainq1 = vectorizer.transform(train_data['Question1'].values.astype('str'))
    trainq2 = vectorizer.transform(train_data['Question2'].values.astype('str'))

    Jmatrix = sparse.csr_matrix((train_data['Jsimilarity']).to_numpy().reshape(-1, 1))
    Fmatrix = sparse.csr_matrix((train_data['Token_set_ratio']).to_numpy().reshape(-1, 1))

    train_set = hstack([trainq1, trainq2])
    #m = hstack([Fmatrix, Jmatrix])
    m = Jmatrix
    normalizer = preprocessing.Normalizer()
    m = normalizer.fit_transform(m)
    train_set = hstack([train_set,m])

    print(train_set.shape)


    # SVM classifier
    clf = LinearSVC(random_state=0, tol=1e-5, C=0.6, loss='hinge', max_iter=100000)
    # Random Forest classifier
    #clf = RandomForestClassifier(n_estimators=100, max_features='auto',max_depth=100,verbose=3,n_jobs=2)

    print("5 fold cv with tfidf")

    scoring = {
        'acc': 'accuracy',
        'prec_macro': 'precision_macro',
        'rec_macro': 'recall_macro',
        'f1_macro': 'f1_macro'

    }

    scores = cross_validate(clf, train_set, train_data['IsDuplicate'], cv=5, scoring=scoring, return_train_score=False,n_jobs=-1,verbose=1)
    print('Accuracy:', np.mean(scores['test_acc']), scores['test_acc'])
    print('Precision:', np.mean(scores['test_prec_macro']), scores['test_prec_macro'])
    print('Recall:', np.mean(scores['test_rec_macro']), scores['test_rec_macro'])
    print('F-Measure:', np.mean(scores['test_f1_macro']), scores['test_f1_macro'])

    #predict test set



