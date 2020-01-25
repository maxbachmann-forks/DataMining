import pandas as pd
from sklearn import preprocessing, svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import cross_val_score


def BoW_SVM(train_data):
    # bow vectorization
    count_vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)
    X = count_vectorizer.fit_transform(train_data['Content'])

    # setting the labels array
    le = preprocessing.LabelEncoder()
    le.fit(train_data["Label"])
    y = le.transform(train_data["Label"])

    # training model
    clf = svm.SVC(kernel='linear')
    clf.fit(X, y)

    # 5 fold cross validation
    scores = cross_val_score(clf, X, y, cv=5)

    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


if __name__ == "__main__":
    # read train_set.csv
    train_data = pd.read_csv('../../dataset/datasets/q1/train.csv', sep=',')
    # train_data = train_data[:500]

    BoW_SVM(train_data)
