import os
from os import path

import numpy as np
import pandas as pd
from PIL import Image
from wordcloud import WordCloud, STOPWORDS


def create_wordcloud(df, currentDir, label, stopwordsList, imgMask):
    data = df.loc[df['Label'] == label]
    content = data['Title']

    # generating wordcloud
    wc = WordCloud(background_color='black', mask=imgMask, stopwords=stopwordsList, max_words=250)
    wc.generate("".join(content))

    wc.to_file(os.path.join(currentDir, label + "_wc.png"))


if __name__ == "__main__":
    # read train_set.csv
    train_set = pd.read_csv('../dataset/datasets/q1/train.csv', sep=',')

    # current directory path
    currentDir = os.path.dirname(__file__);

    # set stopwords and mask
    stopwords = set(STOPWORDS)
    stopwords.update(['says'], ['talk'], ['open'], ['take'], ['see'], ['will'], ['may'], ['new'], ['make'])

    mask = np.array(Image.open(path.join(currentDir, "batman-mask.jpg")))

    # creating wordcloud for each category
    create_wordcloud(train_set, currentDir, "Business", stopwords, mask)
    create_wordcloud(train_set, currentDir, "Entertainment", stopwords, mask)
    create_wordcloud(train_set, currentDir, "Health", stopwords, mask)
    create_wordcloud(train_set, currentDir, "Technology", stopwords, mask)
