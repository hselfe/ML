#!/usr/bin/python
# -*- coding: utf-8 -*-
import requests
import urllib.request
import urllib.parse
import urllib.error
from bs4 import BeautifulSoup
import ssl
import json
import os


class Insta_Info_Scraper:

    def getinfo(self, url):
        html = urllib.request.urlopen(url, context=self.ctx).read()
        soup = BeautifulSoup(html, 'html.parser')
        data = soup.find_all('meta', attrs={'property': 'og:description'
                             })
        text = data[0].get('content').split()
        user = '%s %s %s' % (text[-3], text[-2], text[-1])
        followers = text[0]
        following = text[2]
        posts = text[4]
        f.write ('User:')
        f.write (user + "\r\n")
        f.write ('Followers:')
        f.write( followers + "\r\n")
        f.write ('Following:')
        f.write( following + "\r\n")
        f.write ('Posts:')
        f.write(posts + "\r\n")

        data = json.loads(soup.find('script', type='application/ld+json').text)
        f.write ('Email:')
        try:
            f.write(data['email'] + "\r\n")
        except:
            f.write("NO EMAIL \r\n")
        f.write ('telephone:')
        try:
            f.write(data['telephone'] + "\r\n")
        except:
            f.write("NO TEL \r\n")

        f.write ('---------------------------' + "\r\n")

    def main(self):
        self.ctx = ssl.create_default_context()
        self.ctx.check_hostname = False
        self.ctx.verify_mode = ssl.CERT_NONE

        with open('insta.txt') as f:
            self.content = f.readlines()
        self.content = [x.strip() for x in self.content]
        for url in self.content:
            self.getinfo(url)


if __name__ == '__main__':
    try:
        os.remove("output.txt")
    except:
        f = open("output.txt", "a")
    f = open("output.txt", "a")
    obj = Insta_Info_Scraper()
    obj.main()
