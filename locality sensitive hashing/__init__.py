import pandas as pd
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# def exact_cosine_similarity():
from sklearn.metrics.pairwise import cosine_similarity


def process_data():
    train_data = pd.read_csv('../dataset/datasets/q2a/corpusTrain.csv', sep=',')
    train_data = train_data[:1]

    test_data = pd.read_csv('../dataset/datasets/q2a/corpusTest.csv', sep=',')
    test_data = test_data[:3]

    vectorizer = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS)
    train_data = vectorizer.fit_transform(train_data['Content'])
    test_data = vectorizer.transform(test_data['Content'])

    return train_data, test_data


def exact_cosine(train_data, test_data):
    similarity = cosine_similarity(test_data, train_data, dense_output=False)
    # print(similarity.nonzero())
    # print(similarity.argmax())
    # x, y = similarity.get_shape()
    #
    duplicates = 0
    # for i in range(x):
    #     row = similarity.getrow(i)
    #     for cell in row:
    #         if cell >= threshold:
    #             duplicates += 1
    #             break

    print('Exact Cosine Duplicates:', duplicates)


if __name__ == "__main__":
    threshold = 0.8

    train_data, test_data = process_data()
    exact_cosine(train_data, test_data)
