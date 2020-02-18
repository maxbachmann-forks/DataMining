from collections import defaultdict
import time
import numpy as np
import pandas as pd
import scipy
from datasketch import MinHashLSH, MinHash
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def vectorize_data(train, test):
    stop_words = stopwords.words('english')
    vectorizer = TfidfVectorizer(stop_words=stop_words)
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


def exact_cosine(train, test, threshold):

    query_start = time.time()
    similarity = cosine_similarity(test, train, dense_output=False)
    x, y = similarity.get_shape()
    print(x, y)

    duplicates = 0
    for i in range(x):
        row = similarity.getrow(i)
        row = row.toarray()

        if np.any(row >= threshold):
            duplicates += 1

    query_end = time.time()
    print("Build time: 0 seconds")
    print("Query time: ", query_end - query_start, ' seconds')
    print('Total time: ', query_end - query_start, ' seconds')
    print('Exact Cosine duplicates found:', duplicates)

def exact_jaccard(train, test):

    duplicates=0

    for q1 in tqdm(test['Content']):
        for q2 in train['Content']:
            if  (jaccard_score(q1,q2)) > 0.8:
                duplicates+=1

    print('Exact Jaccard Duplicates:', duplicates)


def lsh_jaccard(train, test, threshold, permutations):

    build_start = time.time()
    lsh = MinHashLSH(threshold=threshold, num_perm=permutations)

    index = 0
    for row in train['Content']:
        x = MinHash(num_perm=permutations)
        for word in row:
            x.update(word.encode('utf8'))

        lsh.insert(index, x)
        index += 1

    build_end = time.time()
    print('Build time: ', build_end - build_start, ' seconds')

    query_start = time.time()
    index = 0
    duplicates = 0
    for row in test['Content']:
        x = MinHash(num_perm=permutations)
        for word in row:
            x.update(word.encode('utf8'))

        result = lsh.query(x)
        if len(lsh.query(x)) > 0:
            duplicates += 1

    query_end = time.time()
    print("Query time: ", query_end - query_start, ' seconds')
    print('Total time: ', (query_end - query_start) + (build_end - build_start), ' seconds')
    print("LSH Jaccard duplicates found:", duplicates)
    print()


def lsh_cosine(train, test, threshold):

    def generate_hyperplanes(dim, k):
        """
        :param dim: number of features
        :param k: number of hyperplanes
        :return: random vector - hyperplane
        """

        return np.random.randn(dim, k)

    def get_powers_of_two(dim):
        """
        this array is useful for finding the index of the table structure
        based on the hyperplanes

        k[1] : 2
        powers_of_two : [2, 1]
        bucket : [0, 1]

        by powers_of_two @ bucket we actually do exponentiation of the bits of the hyperplane with the power 2
        to find the index of the table we are gonna append the points index =>

        for the specific case we have above the bucket is [0, 1] so its index in the hash table is 1
        """

        powers_of_two = []
        powers_of_two.extend(range(0, dim))
        powers_of_two = list(map(lambda x: pow(2, x), powers_of_two))
        powers_of_two.reverse()

        return powers_of_two

    def hash(vector, random_vectors, powers_of_two):
        """
        dot product with the hyperplanes to get the bit array
        and then a dot product of the bit array with the array of powers of two to get
        the exact index of the hash table
        :return: the index hashed
        """

        # find in which bucket the point is hashed
        bucket = vector.dot(random_vectors) >= 0
        bucket = np.where(bucket, 1, 0)

        # calculate the index based on the bucket
        index = bucket @ powers_of_two

        return index[0]

    # Get the dimensions of our set
    features = train.get_shape()[1]

    """
    Concatenating k randomly chosen hash functions
    k : number of vectors
    k = [1, 2, ... , 10]
    """
    k_array = []
    k_array.extend(range(1, 11))

    """
    we run LSH by changing k every time based on the array
    """
    for k in k_array:
        random_vectors = generate_hyperplanes(dim=features, k=k)
        powers_of_two = get_powers_of_two(dim=k)

        # for each index of the table we are gonna save the indices of the train data
        hash_table = defaultdict(list)

        print('k = ', k)
        build_start = time.time()
        i = 0
        for vector in train:
            # hash vector
            index = hash(vector=vector, random_vectors=random_vectors, powers_of_two=powers_of_two)

            # append the index of the vector in the train set to the hash table
            hash_table[index].append(vector)

            i += 1

        hash_table_vectors = []
        for i in range(0, len(hash_table)):
            hash_table_vectors.append(scipy.sparse.vstack(hash_table[i]))

        build_end = time.time()
        print('\tBuild time: ', build_end - build_start, ' seconds')

        testing_start = time.time()
        duplicates = 0
        for vector in test:

            # get the bucket test vector hashes
            index = hash(vector, random_vectors, powers_of_two)

            sim = cosine_similarity(vector, hash_table_vectors[index], dense_output=False)
            if sim.max() >= threshold:
                duplicates += 1

        testing_end = time.time()
        print('\tQuery time: ', testing_end - testing_start, ' seconds')
        print('\tTotal time: ', (testing_end - testing_start) + (build_end - build_start), ' seconds')
        print('\tLSH Cosine Duplicates found: ', duplicates)
        print()


if __name__ == "__main__":
    threshold = 0.8
    permutations = 16

    # reading the train and test set
    train_data = pd.read_csv('../datasets/q2a/corpusTrain.csv', sep=',')
    test_data = pd.read_csv('../datasets/q2a/corpusTest.csv', sep=',')
    # train_data = train_data[:100000]
    # test_data = test_data[:2]

    train_split = train_data
    test_split = test_data

    lsh_jaccard(train=train_data, test=test_data, threshold=threshold, permutations=permutations)

    train_data, test_data = vectorize_data(train=train_data, test=test_data)
    exact_cosine(train=train_data, test=test_data, threshold=threshold)
    # lsh_cosine(train=train_data, test=test_data, threshold=threshold)
    exact_jaccard(train_split, test_split)

