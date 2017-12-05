# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     pca
   Description :
   Author :       yzl
   date：          17-12-4
-------------------------------------------------
   Change Activity:
                   17-12-4:
-------------------------------------------------
"""
import gensim
import scipy
import numpy as np
import pandas
import logging
import os
import re
import sys
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets.samples_generator import make_blobs
'''

log_path = './log/'+time.strftime('%Y-%m-%d',time.localtime(time.time()))+'.log'
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    #filename=log_path,
                    filemode='w'
                    )
'''
def pca(data=None,dimension = -1):
    if data is None:
        print("No input data")
        return
    if dimension == -1:
        dimension = len(data[0])
        print(dimension)
    print("Input data conclude %i rows and %i dimensions."%(len(data),len(data[0])))
    pca = PCA(n_components=dimension)
    newData = pca.fit_transform(data)
    print("Output data conclude %i rows and %i dimensions."%(len(newData),len(newData[0])))
    return newData
if __name__ == '__main__':
    data = [[ 1.  ,  1.  ],
       [ 0.9 ,  0.95],
       [ 1.01,  1.03],
       [ 2.  ,  2.  ],
       [ 2.03,  2.06],
       [ 1.98,  1.89],
       [ 3.  ,  3.  ],
       [ 3.03,  3.05],
       [ 2.89,  3.1 ],
       [ 4.  ,  4.  ],
       [ 4.06,  4.02],
       [ 3.97,  4.01]]
    print(pca(data,1))
