# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     myexp
   Description :
   Author :       yzl
   date：          17-10-19
-------------------------------------------------
   Change Activity:
                   17-10-19:
-------------------------------------------------
"""
import scipy
import numpy
import pandas
import logging
import os
import re
import sys
from mysentences import Mysentences
import word2vec
class Myexp :
    def __init__(self,corpus_dir = None ,sent_log_dir = "./train/sent_log_dir",result_dir = "./train"):
        self.corpus_dir= corpus_dir
        self.sent_log_dir = sent_log_dir
        self.res_catalog = result_dir
    def run(self):
        sentences = Mysentences(self.corpus_dir,self.sent_log_dir)
        #gen = word2vec.Word2Vec(sentences,batch_words=10,hs=1)
        gen = word2vec.Word2Vec(sentences,hs=1)
        #gen = word2vec.Word2Vec(sentences)
        gen.save("./res/huff_beg")
        for _ in range(3):
            gen.finalize_vocab(update=True)
            gen.train(sentences, total_examples=gen.corpus_count, epochs=gen.iter, start_alpha=gen.alpha, end_alpha=gen.min_alpha)
            gen.save("./res/kd_tree"+str(_+1))




