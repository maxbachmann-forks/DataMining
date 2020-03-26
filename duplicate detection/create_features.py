import pandas as pd
from rapidfuzz import fuzz
from tqdm import tqdm


def jaccard_score(q1,q2):
    try:
        q1 = set(q1.lower().split())
        q2 = set(q2.lower().split())

        i = len(q1.intersection(q2))
        u = len(q1) + len(q2) - i

        x = float(i) / float(u)
        return int(x*100)
    except:
        return 0

if __name__ == "__main__":

    train_data = pd.read_csv('../dataset/datasets/q2b/train.csv', sep=',')
    test_data = pd.read_csv('../dataset/datasets/q2b/test_without_labels.csv', sep=',')

    '''
    train_data['Combined'] = train_data['Question1'] + ' ' + train_data['Question2']
    train_data['Token_set_ratio'] = 0
    train_data['Jsimilarity'] = 0

    pd.set_option('display.max_colwidth', 10)

    print 'creating features'

    for index, row in tqdm(train_data.iterrows()):
        train_data.at[index,'Token_set_ratio'] = fuzz.token_set_ratio(row['Question1'], row['Question2'])
        train_data.at[index,'Jsimilarity'] = jaccard_score(row['Question1'], row['Question2'])
    print train_data.head()

    train_data.to_csv('./features',index=False)
    '''
    #create features for test data

    test_data['Combined'] = test_data['Question1'] + ' ' + test_data['Question2']
    test_data['Token_set_ratio'] = 0
    test_data['Jsimilarity'] = 0

    print 'creating features'

    for index, row in tqdm(test_data.iterrows()):
        test_data.at[index, 'Token_set_ratio'] = fuzz.token_set_ratio(row['Question1'], row['Question2'])
        test_data.at[index, 'Jsimilarity'] = jaccard_score(row['Question1'], row['Question2'])
    print test_data.head()

    test_data.to_csv('./features_test', index=False)

