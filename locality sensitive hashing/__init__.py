import numpy as np
import pandas as pd
from datasketch import MinHashLSH, MinHash
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def vectorize_data(train, test):
    vectorizer = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS)
    train = vectorizer.fit_transform(train['Content'])
    test = vectorizer.transform(test['Content'])

    return train, test

def jaccard_score(q1,q2):
    try:
        q1 = set(q1.split())
        q2 = set(q2.split())

        i = len(q1.intersection(q2))
        u = len(q1) + len(q2) - i

        x = float(i) / float(u)
        return x
    except:
        return 0.0


def exact_cosine(train, test):
    similarity = cosine_similarity(test, train, dense_output=False)
    x, y = similarity.get_shape()
    print(x, y)

    duplicates = 0
    for i in range(x):
        row = similarity.getrow(i)
        row = row.toarray()

        if np.any(row >= threshold):
            duplicates += 1

    print('Exact Cosine Duplicates:', duplicates)

def exact_jaccard(train, test):

    duplicates=0

    for q1 in tqdm(test['Content']):
        for q2 in train['Content']:
            if  (jaccard_score(q1,q2)) > 0.8:
                duplicates+=1

    print('Exact Jaccard Duplicates:', duplicates)


def lsh_jaccard(train, test, threshold, permutations):
    lsh = MinHashLSH(threshold=threshold, num_perm=permutations)

    index = 0
    for row in train['Content']:
        x = MinHash(num_perm=permutations)
        for word in row:
            x.update(word.encode('utf8'))

        lsh.insert(index, x)
        index += 1

    index = 0
    duplicates = 0
    for row in test['Content']:
        x = MinHash(num_perm=permutations)
        for word in row:
            x.update(word.encode('utf8'))

        result = lsh.query(x)
        if len(lsh.query(x)) > 0:
            duplicates += 1

    print("LSH Jaccard Duplicates:", duplicates)


if __name__ == "__main__":
    threshold = 0.8

    train_data = pd.read_csv('../dataset/datasets/q2a/corpusTrain.csv', sep=',')
    test_data = pd.read_csv('../dataset/datasets/q2a/corpusTest.csv', sep=',')

    train_split = train_data
    test_split = test_data

    exact_jaccard(train_split, test_split)

