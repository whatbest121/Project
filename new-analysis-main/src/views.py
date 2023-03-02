from flask import Blueprint,render_template,request
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
import heapq
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt  # For Visualisation
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split


views = Blueprint('views',__name__)

# controller
@views.route('/')
def home():
    return render_template("home.html")

@views.route('/sentiment',methods=['GET','POST'] )
def sentiment():
    prediction= ""
    if request.method == 'GET':
        return render_template("sentiment.html",prediction=prediction)
    else:
        input_new = request.form['newInput']
        if input_new != None and input_new!="":
            prediction = predictSentiment(input_new)  
        return render_template("sentiment.html",prediction=prediction)

@views.route('/summary-articles',methods=['GET','POST'] )
def summaryArticles():
    summary = ""
    if request.method == 'GET':
        return render_template("summary-article.html",summary=summary)
    else:
        input_articles = request.form['articles']
        if input_articles != None and input_articles!="":
            summary = summaryArticlesWithInput(input_articles)
        return render_template("summary-article.html",summary=summary)
    

# model
def predictSentiment(newInput):
    data = pd.read_csv('src/static/main.csv')
    data = data[['Dates', 'News', 'PriceSentiment']]
    print(data.shape)
    data.head(7)

    data.isnull().sum()

    data = data.dropna()
    data.isnull().sum()

    # -- NLP --#
    nltk.download('stopwords')

    stemmer = PorterStemmer()
    words = stopwords.words("english")

    data['processedtext'] = data['News'].apply(lambda x: " ".join([stemmer.stem(
        i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
    data.head(10)

    # Pre-process Data
    data['processedtext'] = data['processedtext'].str.strip().str.lower()

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

    # Model Generation
    model = MultinomialNB()
    model.fit(x, y)
    prediction = model.predict(vec.transform([newInput]))  # Output
    prediction =prediction[0]
    return prediction


def summaryArticlesWithInput(article_text):
    # Removing Square Brackets and Extra Spaces
    article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)
    article_text = re.sub(r'\s+', ' ', article_text)
    # Removing special characters and digits
    formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text )
    formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)
    sentence_list = nltk.sent_tokenize(article_text)
    stopwords = nltk.corpus.stopwords.words('english')

    word_frequencies = {}
    for word in nltk.word_tokenize(formatted_article_text):
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
    maximum_frequncy = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)
    sentence_scores = {}
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]

    summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(summary_sentences)
    return summary


