from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError, CollectionInvalid
from bs4 import BeautifulSoup
from PIL import Image
import requests
import time


def single_query(link, payload):
    response = requests.get(link, params=payload)
    if response.status_code != 200:
        print 'WARNING', response.status_code
    else:
        return response.json()






def loop_through(total_pages,pay_load,link,tab):
    for i in xrange(total_pages):
        if i %50 ==0:
            print 'page {}'.format(i)
        pay_load['page'] = str(i)
        content = single_query(link,pay_load)
        print content
        meta_lst = content['response']['docs']

        for meta in meta_lst:
            try:
                tab.insert_one(meta)
            except DuplicateKeyError:
                print "Duplicates"
'''UNFINISHED'''

if __name__ == '__main__':
    link = 'http://api.nytimes.com/svc/search/v2/articlesearch.json'
    pay_load = {'api-key': 'fd7bb2c9fd2a440abe8963b1d0ebec94'}
    client = MongoClient()
    # Access/Initiate Database
    db = client['nytimes']
    # Access/Initiate Table
    tab = db['nyt']

    loop_through(100,pay_load,link,tab)
