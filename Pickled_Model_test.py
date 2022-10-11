# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 22:40:25 2021

@author: thiir
"""

import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from PIL import Image
import skimage
import sklearn
import numpy as np
from skimage.feature import hog
from skimage import data, color, exposure
from sklearn import svm
import glob
import os
import mahotas
import imutils
from scipy import misc
import scipy
from skimage import io
import pickle

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

# load the model from disk
clf = pickle.load(open('finalized_model.sav', 'rb'))

files=glob.glob('*.jpg')


for file in files:




    #predfile='61.07 Kmh  16h-20m-12s.bmpPRD.png'
    predfile='car_at_20210717_132615.png'
    img=plt.imread(predfile)
    grayscale = rgb2gray(img)
    fdPred = hog(grayscale, orientations=8, pixels_per_cell=(6, 6),cells_per_block=(3, 3), visualize=False)



    prob=clf.predict_proba([fdPred])
    if clf.predict([fdPred])==1:
        print('The Vehicle is an SUV')
        v_type='SUV'
        plt.savefig('file'+ v_type +'.png')
        
    if clf.predict([fdPred])==2:
        print('The Vehicle is a Sedan')
        v_type='SED'
        plt.savefig('file'+ v_type +'.png')
        
    if clf.predict([fdPred])==3:
        print('The Vehicle is aTruck')
        v_type='TRK'
        plt.savefig('file'+ v_type +'.png')


