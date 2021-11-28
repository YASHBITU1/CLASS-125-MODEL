from sys import version
import numpy as np
from numpy.random.mtrand import multinomial 
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps

X = np.load('image.npz')['arr_0']
y = pd.read_csv("lables.csv")["labels"]

print(pd.Series(y) .value_counts())

classes = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','X','Y','Z']

nclasses = len(classes)

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 9,train_size=3500,test_size = 500)
X_train_scaled = X_train / 255.0
X_test_scaled = X_test / 255.0
clf = LogisticRegression(solver="saga",multi_class="multinomial").fit(X_train_scaled,y_train)

def getPrediction(image):
    im_open = Image.open(image)
    im_bw = im_open.convert("L")
    im_bwrz = im_bw.resize((22,30),Image.ANTIALIAS)
    pixelFilter = 20
    minPixel = np.percentile(im_bwrz,pixelFilter)
    im_bwrzinv = np.clip(im_bwrz-minPixel,0,255)
    maxPixel = np.max(im_bwrz)
    im_bwrzinv = np.asarray(im_bwrzinv)/maxPixel
    testSample = np.array(im_bwrzinv).reshape(1,784)
    testPredict = clf.predict(testSample)
    return testPredict[0]

    


