import pandas as pd
import numpy as np
from sklearn import preprocessing, svm
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import cross_val_score, cross_validate


def SVM(train_data, SVD):
    # bow vectorization
    count_vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)
    X = count_vectorizer.fit_transform(train_data['Content'])

    if SVD:
        svd = TruncatedSVD(n_components=10, random_state=1)
        X = svd.fit_transform(X)

    # setting the labels array
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(train_data["Label"])

    # SVM classifier
    clf = svm.SVC(kernel='linear')

    # 5 fold cross validation
    print("Attempting 5-fold cross validation...")
    scores = cross_val_score(clf, X, y, cv=5)

    scoring = {
        'acc': 'accuracy',
        'prec_macro': 'precision_macro',
       'rec_macro': 'recall_macro',
        'f1_macro': 'f1_macro'

    }
    scores = cross_validate(clf, X, y, cv=5, scoring=scoring, return_train_score=False)
    print('Accuracy:',  np.mean(scores['test_acc']))
    print('Precision:', np.mean(scores['test_prec_macro']))
    print('Recall:',    np.mean(scores['test_rec_macro']))
    print('F-Measure:', np.mean(scores['test_f1_macro']))

    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


if __name__ == "__main__":
    # read train_set.csv
    train_data = pd.read_csv('../../dataset/datasets/q1/train.csv', sep=',')
    # train_data = train_data[:500]

    SVM(train_data, SVD=False)
    SVM(train_data, SVD=True)
