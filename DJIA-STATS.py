import pandas as pd
import numpy as np
import datetime as dt
import os
import re
# import sklearn ...

if __name__ == '__main__':

    # Load data 
    news = pd.read_csv(os.path.join('Data','News_DJIA.csv'))
    price = pd.read_csv(os.path.join('Data','Value_DJIA.csv'))
    news = news.merge(price, on='Date')
    
    # combine titles 
    title_cols = list(news.columns[2:24]) # using Top 22 titles due to BERTs max sequence length of 512
    news['News'] = news[title_cols].agg(' '.join, axis = 1)

    # remove, quotes and b, from news title cols 
    def clean_titles(titles):
        titles = re.sub('b[(\')]','',titles)
        titles = re.sub('b[(\")]','',titles)
        titles = re.sub("\'",'',titles)
        return titles

    news['News'] = news.apply(lambda x: clean_titles(x['News']), axis = 1)

    # drop un-used cols
    news = news.drop(news.columns[2:27], axis = 1)

    # Date to datetime object
    news['Date'] = pd.to_datetime(news['Date'])

    # split news to train and test
    split = dt.datetime(2015,1,1,0,0,0)
    train = news[news.Date <= split]
    test = news[news.Date > split]
    X_train, y_train = np.array(train['News']),np.array(train['Label'])
    X_test, y_test = np.array(test['News']),np.array(test['Label'])