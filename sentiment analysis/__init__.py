import pandas as pd
import numpy as np
from sklearn import preprocessing, svm
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.tree import DecisionTreeClassifier


def SVM_cross_val():

    # SVM classifier
    clf = LinearSVC()

    # 5 fold cross validation
    print("Attempting 5-fold cross validation...")

    scoring = {
        'acc': 'accuracy',
        'prec_macro': 'precision_macro',
        'rec_macro': 'recall_macro',
        'f1_macro': 'f1_macro'

    }
    scores = cross_validate(clf, X, y, cv=5, scoring=scoring, return_train_score=False, n_jobs=-1)
    print('Accuracy:', np.mean(scores['test_acc']))
    print('Precision:', np.mean(scores['test_prec_macro']))
    print('Recall:', np.mean(scores['test_rec_macro']))
    print('F-Measure:', np.mean(scores['test_f1_macro']))


def SVM():

    # train SVM
    clf = LinearSVC()
    clf.fit(X, y)

    return clf.predict(X_test)


if __name__ == "__main__":

    # read train_set.csv
    train_data = pd.read_csv('../datasets/q3/train.csv', sep=',')
    test_data = pd.read_csv('../datasets/q3/test_without_labels.csv', sep=',')

    # process data
    tfidf_vectorizer = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS)
    X = tfidf_vectorizer.fit_transform(train_data['Content'])
    X_test = tfidf_vectorizer.transform(test_data['Content'])

    # setting the labels array
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(train_data["Label"])

    SVM_cross_val()

    y_pred = SVM()
    prediction = pd.DataFrame(data={"Predicted": y_pred}, index=test_data['Id'])
    prediction.to_csv('sentiment_predictions.csv')
