from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import pandas as pd
import numpy as np
import datetime as dt
import os
import re
# import sklearn ...

# takes text block and returns sentiment related results
sia = SentimentIntensityAnalyzer()
def get_sentiment_vals(text):

    blob = TextBlob(text)
    subjectivity = blob.sentiment.subjectivity
    polarity = blob.sentiment.polarity

    sentiment = sia.polarity_scores(text)
    compound = sentiment['compound']
    pos = sentiment['pos']
    neu = sentiment['neu']
    neg = sentiment['neg']

    return [subjectivity,polarity,compound,pos,neu,neg]

# remove, quotes and b, from news text block
def clean_titles(titles):
    titles = re.sub('b[(\')]','',titles)
    titles = re.sub('b[(\")]','',titles)
    titles = re.sub("\'",'',titles)
    return titles


if __name__ == '__main__':

    # Load data 
    news = pd.read_csv(os.path.join('Data','News_DJIA.csv'))
    price = pd.read_csv(os.path.join('Data','Value_DJIA.csv'))
    
    # combine titles 
    title_cols = list(news.columns[2:24]) # using Top 22 titles due to BERTs max sequence length of 512
    news['News'] = news[title_cols].agg(' '.join, axis = 1)

    # remove, quotes and b, from news title cols 
    news['News'] = news.apply(lambda x: clean_titles(x['News']), axis = 1)

    # drop un-used cols
    news = news.drop(news.columns[2:27], axis = 1)


    # Add sentiment
    news[['subjectivity','polarity','compound','pos','neu','neg']] = news.apply(lambda x: pd.Series(get_sentiment_vals(x['News'])),axis = 1)
    
    # merge price to news
    data = news.merge(price, on='Date')
    
    # Date to datetime object
    data['Date'] = pd.to_datetime(data['Date'])

    # split news to train and test
    split = dt.datetime(2015,1,1,0,0,0)
    train = data[data.Date <= split]
    test = data[data.Date > split]
    x_cols = ['Open','Close','High','Low','News']
    X_train, y_train = np.array(train[x_cols]),np.array(train['Label'])
    X_test, y_test = np.array(test[x_cols]),np.array(test['Label'])