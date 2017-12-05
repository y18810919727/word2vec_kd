# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     mysentences
   Description :
   Author :       yzl
   date：          17-10-19
-------------------------------------------------
   Change Activity:
                   17-10-19:
-------------------------------------------------
"""
import gensim
import scipy
import numpy
import pandas
import logging
import os
import codecs
import nltk
import re
import time
import sys
class Mysentences:
    def __init__(self,data_dir,):
        self.dirname = data_dir
        self.sent_cnt=0;
    def clean_html(self,raw):
        cleanr = re.compile('<.*?>')
        text = re.sub(cleanr,' ',raw)
        return text
    def __iter__(self):
        for root,dirs,files in os.walk(self.dirname):
            for d in files:
                filepath = os.path.join(root,d)
                try:
                    the_file = open(filepath)
                    for line in the_file:
                        sl = line.strip()
                        rline = self.clean_html(sl)
                        if sl=="":
                            continue
                        try:
                            token_line = ' '.join(nltk.word_tokenize(rline))
                        except UnicodeDecodeError as err:
                            print(filepath)
                            print(err)
                            exit()
                        word_line = [word for word in
                                     token_line.lower().split()
                                     if word.isalpha()]
                        if(len(word_line)==0):
                            continue
                        #self.wirte_log(" ".join(word_line))
                        yield word_line
                except UnicodeDecodeError as e:
                    print(e)
                    print("Decode Error in %s!!!!", filepath)
                    continue

if __name__ == '__main__':
    sentences = Mysentences("../data/test")
    i=1
    for s in sentences:
        print(i,'.',s)
        i=i+1
