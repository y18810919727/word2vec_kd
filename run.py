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
import BASE_ROOT as ROOT
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
    Data_path = os.path.join(ROOT.BASE_ROOT_DIR,"data")
    Res_path = os.path.join(ROOT.BASE_ROOT_DIR,"res/")
    ev_path = os.path.join(ROOT.BASE_ROOT_DIR, 'code/res/evaluation'+time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))+'.out')
    exp = Myexp(Data_path,Res_path)
    exp.run_exp()
    ev.evaluate_models(Res_path, ev_path);


    #gen = gensim.models.Word2Vec()
