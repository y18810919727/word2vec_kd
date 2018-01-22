# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     new_evaluation
   Description :
   Author :       yzl
   date：          17-12-13
-------------------------------------------------
   Change Activity:
                   17-12-13:
-------------------------------------------------
"""
import gensim
import scipy
import numpy
import pandas
import logging
import os
import re
import sys
import sys
import time
import word2vec
#start

def change_stdout(filename):
    output = sys.stdout
    outputfile =open(filename,'w')
    sys.stdout = outputfile
    return output

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

def run(model,data):
    right_item = 0
    cnt = 0
    cur_right = 0
    cur_sum = 0
    res = []
    for id,x in enumerate(data):
        if x[0]==':':
            if len(x)<=1:
                continue
            if x[1] == 'semantic' :
                cur_right = 0
                cur_sum = 0
            elif x[1] == 'syntactic':
                res.append(1.0*cur_right/cur_sum)
                cur_right = 0
                cur_sum = 0
            continue
        else:
            try:
                if model.wv.most_similar(positive=[x[1], x[2]], negative=[x[0]],topn=1)[0][0] == x[3]:
                    cur_right +=1
                    right_item += 1
            except KeyError as e:
                logging.info(e)
        cur_sum +=1
        cnt += 1
    res.append(1.0*cur_right/cur_sum)
    res.append(1.0*right_item/cnt)
    return res

def evaluate_models(dir_name,output_name):


    output = sys.stdout
    outputfile =open(output_name,'w')
    sys.stdout = outputfile

    data = Data('./toolkit/word-test.v1.txt')
    #data = Data('./toolkit/word-test.v2.txt')
    for dir in os.listdir(dir_name):
        print('modelname :',dir)
        information_file = open(dir_name+'/'+dir+'/information','r')
        for x in information_file.readlines():
            print(x,end='')
        information_file.close()
        model_name = dir_name+'/'+dir+'/model/' + dir
        model = word2vec.Word2Vec.load(model_name)
        result = run(model,data)
        print('semantic : '+str(result[0]))
        print('syntactic : '+str(result[1]))
        print('Total : '+str(result[2]))
        print('\n')
        del model

    sys.stdout = output
    outputfile.close()

if __name__ == '__main__':
    evaluate_models('../res','./res/evaluation'+time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))+'.out');

