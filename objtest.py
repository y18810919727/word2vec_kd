# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     objtest
   Description :
   Author :       yzl
   date：          17-11-20
-------------------------------------------------
   Change Activity:
                   17-11-20:
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

def add(a):
    a+=1
def srt(f):
    print(id(f))
    f.sort()
x=[4,3,2,1]
print(id(x))
srt(x[1:3])
print(x)

