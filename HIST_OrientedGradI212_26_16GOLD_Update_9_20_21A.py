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
#import pdb; pdb.set_trace()
#from pdb import set_trace as bp

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

#SUPERVISED LEARNING USING HISTOGRAM ORIENTED GRADIENTS.
# This code reads in a series of .png image files of different kinds of vehicular traffic outside my office
#window.  The image file names have labels embedded in the file names.  The code reads the images
#in and converts them to grayscale.  Then the code converts the images to Histogram Oriented Gradients.
#the HOG is then used to classify the different HOG based on their labels.  The training set provided
#with this code contains 99 samples of Trucks, Sedans, and Sport Utility Vehicles (SUV).   The code uses
#a debugger pdb() function to step through the code. Simply type c to contine the execution.  To remove
#the debugger stops simply comment out the bp() statements.

#At the very end of the program, an image of an SUV is provided that was not part of the training set.
#The code predicts the probability of the image being a Sedan, Truck or SUV.
#This code was authored by Russell A. Crook All rights researved 2016.
#This code is provided without warranty and is open for use by anyone provide credit is given.

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])



files=glob.glob('*.png')
print (files)
print (len(files))
#http://www.astro.washington.edu/users/vanderplas/Astr599/notebooks/17_SklearnIntro
#bp()
#CountNEG=0
CountTRK=0
CountSED=0
CountSUV=0
labels=[]
label=[]
feature=[]
targets=[]
list_hog_fd=[]

#image size 432 x 654
#dtype float32

for file in files:    
    #bp()
    print (file)
    b=file[-7:-4]
    print ('b=',b)
    if b=='SUV':
        #images of SUV's labels [1,0,0]
        label='SUV'
        #labels=label.append(label)
        y=1
        targets.append(y)
#http://stackoverflow.com/questions/23008447/classification-test-in-scikit-learn-valueerror-setting-an-array-element-with-a 
        #image=misc.imread(file)
        #grayscale=scipy.ndimage.imread(file, flatten=True, mode=None)
        img=plt.imread(file)
        grayscale = rgb2gray(img)
        fd = hog(grayscale, orientations=8, pixels_per_cell=(6, 6),cells_per_block=(3, 3), visualize=False)
        (h,w)=grayscale.shape
        type=grayscale.dtype
        #feature+=[fd]
        list_hog_fd.append(fd)
        #print(feature)
        CountSUV+=1
        print('SUV count=',CountSUV)
        #bp()    
    if b=='SED':
        #images of Sedans labels [0,1,0]
        label='SED'
        #labels=label.append(label)
        y=2
        targets.append(y)
        #image=misc.imread(file)
        #grayscale=scipy.ndimage.imread(file, flatten=True, mode=None)
        img=plt.imread(file)
        grayscale = rgb2gray(img)
        fd= hog(grayscale, orientations=8, pixels_per_cell=(6, 6),cells_per_block=(3, 3), visualize=False)
        (h,w)=grayscale.shape
        type=grayscale.dtype
        #feature+=[fd]
        list_hog_fd.append(fd)
        CountSED+=1
        print('Sedan Count=',CountSED)
    if b=='TRK':
        #images of Trucks labels [0,0,1]
        label='TRK'
        #labels=label.append(label)
        y=3
        targets.append(y)
        #labels+=[y]
        #image=misc.imread(file)
        img=plt.imread(file)
        #grayscale=scipy.ndimage.imread(file, flatten=True, mode=None)'
        img=plt.imread(file)
        grayscale = rgb2gray(img)
        fd= hog(grayscale, orientations=8, pixels_per_cell=(6, 6),cells_per_block=(3, 3), visualize=False)
        (h,w)=grayscale.shape
        type=grayscale.dtype
        #feature+=[fd]
        list_hog_fd.append(fd)
        #print(feature)
        CountTRK+=1
        print('Truck Count=',CountTRK)
feature=np.array(list_hog_fd,'float64')
print ('number of Trucks', CountTRK)
print ('number of Sedans', CountSED)
print ('number of SUVs', CountSUV)
print ('length of features', len(feature))
print ('length of targets', len(targets))

#print label
#print feature
#X=np.asarray(feature)
X=feature
#print (X[0])
clf=svm.SVC(kernel='linear',probability=True)
clf.fit(X,targets)
# save model
filename = 'finalized_model.sav'
pickle.dump(clf, open(filename, 'wb'))





#bp()
print ('Program Complete')
# Making a prediction
print('Now making prediction based on HOG SVM Classifier')
predfile='61.07 Kmh  16h-20m-12s.bmpPRD.png'
img=plt.imread(predfile)
grayscale = rgb2gray(img)
fdPred = hog(grayscale, orientations=8, pixels_per_cell=(6, 6),cells_per_block=(3, 3), visualize=False)
#bp()
prob=clf.predict_proba([fdPred])
#print(prob)
#bp()
print('Probability SUV',prob[0,0])
print('Probability Sedan',prob[0,1])
print('Probability Truck',prob[0,2])
#print(clf.predict(fdPred))
if clf.predict([fdPred])==1:
    print('The Vehicle is an SUV')
if clf.predict([fdPred])==2:
    print('The Vehicle is a Sedan')
if clf.predict([fdPred])==3:
    print('The Vehicle is aTruck')

    

#from sklearn import decomposition
#Creates a Plot of the Principal Components Analysis
#pca=decomposition.PCA(n_components=2)
#pca.fit(X)
#X1=pca.transform(X)
#import pylab as pl
#pl.scatter(X1[:,0],X1[:,1],c=targets)
#pl.legend()
#pl.show()
