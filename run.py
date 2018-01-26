# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     run
   Description :
   Author :       yzl
   date：          17-10-19
-------------------------------------------------
   Change Activity:
                   17-10-19:
-------------------------------------------------
"""
import numpy
import logging
import os
import re
import sys
import time
import new_evaluation as ev
from myexp import  Myexp
log_path = './log/'+time.strftime('%Y-%m-%d-%H-%M',time.localtime(time.time()))+'.log'
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename=log_path,
                    filemode='w'
                    )
if __name__ == '__main__':
    exp = Myexp("../data",'../res/')
    exp.run_exp()
    ev.evaluate_models('../res', './res/evaluation'+time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))+'.out');


    #gen = gensim.models.Word2Vec()
