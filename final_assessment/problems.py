'''
Section 1 of Final Assessment.
You can run tests for this section by running "make test" from the root repo(final-assessment)
'''

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import requests


def column_averages(filename):
    '''
    INPUT: string
    OUTPUT: dictionary (string => float)

    Take the name of a csv file. The first line of the file is the column names.
    All of the column values are numerical. Return a dictionary containing the
    average value for each column.
    '''
    df = pd.read_csv(filename)
    average_d={}

    for col in df.columns:
        average_d[col] = df[col].mean()
    return average_d



def filter_by_class(X, y, label):
    '''
    INPUT: 2 dimensional numpy array, numpy array, object
    OUTPUT: 2 dimensional numpy array

    Return the rows from X whose corresponding label from y is the given label.
    '''
    return X[y==label,:]


def etsy_query(query):
    url = 'https://www.etsy.com/search?q=' +str(query) +'&ref=auto2'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    all_res=soup.findAll(class_='card-meta-row-item text-truncate overflow-hidden card-shop-name')
    usernames = [str(x).split('\n')[1].strip() for x in all_res]
    return usernames
    '''
    INPUT: string
    OUTPUT: list of strings

    Take a query and return a list of all of the etsy usernames who have a
    result on the first page of that query result.
    '''
