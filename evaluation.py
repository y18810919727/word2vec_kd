# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     evaluation
   Description :
   Author :       yzl
   date：          17-11-26
-------------------------------------------------
   Change Activity:
                   17-11-26:
-------------------------------------------------
"""
import gensim
import scipy
import numpy
import pandas
import logging
import time
import os
import re
import sys

log_path = './log/evaluation'+time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))+'.log'
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename=log_path,
                    filemode='w'
                    )
import word2vec
data_root = '/media/yzl/1488-8A10/kd_training/res/huff_res'
#model_path=['../res/kd_tree1','../res/kd_tree2','../res/kd_tree3','../res/huff_beg',data_root+'/huff_tree1',data_root+'/huff_tree2',data_root+'/huff_tree3']
model_path=[data_root+'/huff_beg',data_root+'/huff_tree1',data_root+'/huff_tree2',data_root+'/huff_tree3']

class Data:
    def __init__(self,data_path):
        self.data_path = data_path
    def __iter__(self):
        f = open(self.data_path)
        for line in f:
            if line.find('//') == 0:
                continue
            line = line.replace('\r','').replace('\n','').replace('\t','').lower()
            tup = tuple(line.split(' '))
            yield tup
        f.close()

def read_data(data_path):
    f = open(data_path)
    data = []
    for line in f:
        if line.find(':') != -1:
            continue
        line = line.replace('\r','').replace('\n','').replace('\t','').lower()
        tup = tuple(line.split(' '))
        data.append(tup)
    f.close()
    return data
def evaluate_model(model,data):
    right_item = 0
    cnt = 0
    cur_right = 0
    cur_sum = 0
    for id,x in enumerate(data):
        #print(id)
        if x[0]==':':
            if cur_sum !=0 :
                logging.info("In this group.Sum\t%i Right\t%i Accuracy\t%f",cur_sum,cur_right,(cur_right)/(cur_sum))
            logging.info("Group Name:\t%s",x[1])
            cur_right = 0
            cur_sum = 0
            continue
        if model.wv.most_similar(positive=[x[1], x[2]], negative=[x[0]],topn=1)[0][0] == x[3]:
            cur_right +=1
            right_item += 1

        cur_sum +=1
        cnt += 1
    logging.info("In this group.Sum\t%i Right\t%i Accuracy\t%f",cur_sum,cur_right,(cur_right)/(cur_sum))
    logging.info("Number is %i cnt. The right count is %i.The whole accuracy is %f.",cnt,right_item,(right_item)/(cnt))
    return right_item/cnt

if __name__ == '__main__':
    #model =  word2vec.Word2Vec.load('../res/huff_beg')
    data = Data('../toolkit/word-test.v1.txt')
    for x in model_path:
        model =  word2vec.Word2Vec.load(x)
        res = evaluate_model(model,data)
    #print(len(data))
    #print(model.wv['computer'])