from pywebio.output import put_text
from pywebio.input import input, FLOAT, TEXT
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
import pandas as pd
import csv
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np  # linear algebra
# data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
import matplotlib.pyplot as plt  # For Visualisation
import seaborn as sns  # For better Visualisation
from bs4 import BeautifulSoup  # For Text Parsing

import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('main.csv')
data = data[['Dates', 'News', 'PriceSentiment']]
print(data.shape)
data.head(7)

data.isnull().sum()

data = data.dropna()
data.isnull().sum()

Sentiment = data['PriceSentiment'].unique()
print(Sentiment)

data.groupby(data['PriceSentiment']).News.count().plot.bar(ylim=0)
plt.show()


# -- NLP --#
nltk.download('stopwords')

stemmer = PorterStemmer()
words = stopwords.words("english")

data['processedtext'] = data['News'].apply(lambda x: " ".join([stemmer.stem(
    i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
print(data.shape)
data.head(10)

# Pre-process Data


def preprocess_data(data):
    data['processedtext'] = data['processedtext'].str.strip().str.lower()
    return data


data = preprocess_data(data)

# Splitting Data
df = data
# Split into training and testing data
x = data['News']
y = data['PriceSentiment']
x, x_test, y, y_test = train_test_split(
    x, y, stratify=y, test_size=0.3, random_state=42)
# Vectorize text reviews to numbers
vec = CountVectorizer(stop_words='english')
x = vec.fit_transform(x).toarray()
x_test = vec.transform(x_test).toarray()

## pywebio##


def main():
    # Model Generation
    from sklearn.naive_bayes import MultinomialNB

    model = MultinomialNB()
    model.fit(x, y)
    from itertools import count
    import pandas as pd
    df = pd.read_csv('gold-dataset-sinha-khandait.csv', sep=',', header=None)
    start = 10000
    end = 10570
    df = df[start - 1:end - 1]
    correct = 0
    str = input('This is Text', type=TEXT, placeholder='News forex Gold sport',
                help_text='This is help text', required=True)
    put_text(model.predict(vec.transform([str])))  # Output
    for i in range(len(df)):
        print(df.values[i][2])
        put_text(model.predict(vec.transform([df.values[i][2]])), df.values[i][9] == model.predict(
            vec.transform([df.values[i][2]])))  # Output

        if df.values[i][9] == model.predict(vec.transform([df.values[i][2]])):
            correct += 1

    print(correct / len(df) * 100)


if __name__ == '__main__':
    main()
