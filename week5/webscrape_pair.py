from pymongo import MongoClient
from bs4 import BeautifulSoup
from PIL import Image
import requests


client = MongoClient()
# Access/Initiate Database
db = client['test_database']
# Access/Initiate Table
tab = db['test_table']

#3
#class='sresult lvresult'

#class = lvpicinner full-width picW

# f = open("/Users/gschoolstudent/Desktop/webscrape/ebay_shoes.html")

# soup = BeautifulSoup(f, 'html.parser')
#
# result = soup.select('.lvpicinner.full-width.picW')
# images =[]
# #for each in result:
#     images.append( each.find('img')['src'])
#
# #for i, each_img in enumerate(images):
#     each_img = each_img[0:]
#     # fil = open('/Users/gschoolstudent/Desktop/webscrape','w')
#     im = Image.open(each_img)
#     im.save('/Users/gschoolstudent/Desktop/webscrape/' + str(i)+".jpeg","JPEG")

url = 'http://www.ebay.com/sch/i.html?_from=R40&_trksid=p2050601.m570.l1313.TR2.TRC0.A0.H0.Xdata+science.TRS0&_nkw=data+science&_sacat=0'

r = requests.get(url)
soup = BeautifulSoup(r.content, 'html.parser')

result = soup.select('.vip')

for each_res in result:
    title = each_res.get_text()
    print title
