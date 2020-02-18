import pandas as pd
from fuzzywuzzy import fuzz
from tqdm import tqdm


def jaccard_score(q1,q2):
    try:
        q1 = set(q1.lower().split())
        q2 = set(q2.lower().split())

        x = float(len(q1 & q2)) / float(len(q1 | q2))
        return int(x*100)
    except:
        return 0

if __name__ == "__main__":

    train_data = pd.read_csv('../dataset/datasets/q2b/train.csv', sep=',')
    test_data = pd.read_csv('../dataset/datasets/q2b/test_without_labels.csv', sep=',')

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


