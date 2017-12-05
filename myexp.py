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
import copy
import time
import os
import re
import sys
from mysentences import Mysentences
import word2vec
class Myexp :
    def __init__(self,corpus_dir=None,res_dir="../res/"):
        self.corpus_dir = corpus_dir
        self.sentences = Mysentences(self.corpus_dir)
        self.res_dir = res_dir

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

    def run2(self):
        sentences = Mysentences(self.corpus_dir,self.sent_log_dir)
        gen = word2vec.Word2Vec(sentences,hs=1)
        #gen = word2vec.Word2Vec(sentences)
        #gen.save("./res/huff_beg")
        gen.finalize_vocab(update=True,kd_tree=True,pca=True)
        gen.train(sentences, total_examples=gen.corpus_count, epochs=gen.iter, start_alpha=gen.alpha, end_alpha=gen.min_alpha)

    def write_model_information(self,information_file,infor_dic):
        file = open(information_file,'w')
        for key in infor_dic:
            file.write(key)
            file.write("\t")
            file.write(str(infor_dic[key]))
            file.write("\n")
        file.close()

    def make_dir(self,dir):
        if not os.path.exists(dir):
            os.mkdir(dir)
    def train_model(self, hs=0, negative=5, sg=0, size=100, pca=True,
                    kd_tree=True,model_name='default', origin_model=None, save_dir='../res/'):
        print("Training ",model_name)
        save_dir = save_dir + model_name
        self.make_dir(save_dir)
        self.make_dir(save_dir+'/model')

        start_time = time.clock()
        '''
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                            datefmt='%a, %d %b %Y %H:%M:%S',
                            filename=save_dir+'/_'+model_name+time.strftime('%Y-%m-%d',time.localtime(time.time()))+'.log',
                            filemode='w'
                            )
        '''
        information_dic = {'hs': hs,'negative': negative, 'sg': sg, 'size': size, 'pca': pca, 'kd_tree': kd_tree,
                           'model_name': model_name}
        if kd_tree:
            origin_model.negative = 0
            origin_model.hs = 1
            origin_model.finalize_vocab(update=True,kd_tree=kd_tree,pca=pca)
            origin_model.train(self.sentences, total_examples=origin_model.corpus_count,
                               epochs=origin_model.iter, start_alpha=origin_model.alpha, end_alpha=origin_model.min_alpha)
            information_dic['train_time'] = time.clock()-start_time
            self.write_model_information(save_dir+'/'+'information', infor_dic=information_dic)
            origin_model.save(save_dir+'/model/'+model_name)
            return origin_model
        else:
            new_model = word2vec.Word2Vec(sentences=self.sentences, hs=hs, negative=negative, sg=sg, size=size)
            information_dic['train_time'] = time.clock()-start_time
            self.write_model_information(save_dir+'/'+'information',infor_dic=information_dic)
            new_model.save(save_dir+'/model/'+model_name)
            return new_model

    def run_exp(self):
        model1 = self.train_model(hs=1,negative=0,sg=0,size=100,pca=False,kd_tree=False,model_name='huf_100_cbow',save_dir=self.res_dir)
        self.train_model(hs=1,negative=0,sg=0,size=100,pca=False,kd_tree=True,model_name='kd_100_cbow',origin_model=copy.deepcopy(model1),save_dir=self.res_dir)
        self.train_model(hs=1,negative=0,sg=0,size=100,pca=True,kd_tree=True,model_name='pca_100_cbow',origin_model=model1,save_dir=self.res_dir)
        model1 = self.train_model(hs=1,negative=0,sg=1,size=100,pca=False,kd_tree=False,model_name='huf_100_sg',save_dir=self.res_dir)
        self.train_model(hs=1,negative=0,sg=1,size=100,pca=False,kd_tree=True,model_name='kd_100_sg',origin_model=copy.deepcopy(model1),save_dir=self.res_dir)
        self.train_model(hs=1,negative=0,sg=1,size=100,pca=True,kd_tree=True,model_name='pca_100_sg',origin_model=model1,save_dir=self.res_dir)
        model1 = self.train_model(hs=1,negative=0,sg=0,size=300,pca=False,kd_tree=False,model_name='huf_300_cbow',save_dir=self.res_dir)
        self.train_model(hs=1,negative=0,sg=0,size=300,pca=False,kd_tree=True,model_name='kd_300_cbow',origin_model=copy.deepcopy(model1),save_dir=self.res_dir)
        self.train_model(hs=1,negative=0,sg=0,size=300,pca=True,kd_tree=True,model_name='pca_300_cbow',origin_model=model1,save_dir=self.res_dir)
        model1 = self.train_model(hs=1,negative=0,sg=1,size=300,pca=False,kd_tree=False,model_name='huf_300_sg',save_dir=self.res_dir)
        self.train_model(hs=1,negative=0,sg=1,size=300,pca=False,kd_tree=True,model_name='kd_300_sg',origin_model=copy.deepcopy(model1),save_dir=self.res_dir)
        self.train_model(hs=1,negative=0,sg=1,size=300,pca=True,kd_tree=True,model_name='pca_300_sg',origin_model=model1,save_dir=self.res_dir)
        self.train_model(hs=0,negative=5,sg=1,size=300,pca=False,kd_tree=False,model_name='neg5_300_sg',save_dir=self.res_dir)








