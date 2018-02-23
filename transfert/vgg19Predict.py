# coding: utf-8
import numpy as np
import pandas as pd
import os
import cv2
import time
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.utils import to_categorical
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import hamming_loss
from keras.applications.vgg19 import VGG19


DIRECT_ORIGINALS='../../images/'
DIRECT_AUGMENTED='../../imagesAugmented/'


batch_size=8

img_width, img_height = 224, 224
NUMBER_OF_DESEASES=1
input_shape = (img_width, img_height,3)
#cropping dimension
crop_x,crop_y,crop_w,crop_h=(112,112,800,800)
np.random.seed(1234)

#get input data
PATHOLOGY_NAME = raw_input("Please enter pathology name : ")
if len(PATHOLOGY_NAME)==0:
    PATHOLOGY_NAME='Cardiomegaly'
    
IMAGE_NAME = raw_input("Please enter image name or 'test' for full validation: ")
if len(IMAGE_NAME)==0:
    IMAGE_NAME='test'
    
MODEL_NAME = raw_input("Please enter model name : ")
if len(MODEL_NAME)==0:
    MODEL_NAME="my"+PATHOLOGY_NAME+".h5"

FILE_NAME='Data_'+PATHOLOGY_NAME+'.csv'

#import image data set description
df = pd.read_csv(FILE_NAME)

#loadModel : create or load already trained model
def loadModel():
    if os.path.exists(MODEL_NAME):
        model = load_model(MODEL_NAME)
    return model


#Create dataset for validation
def loadDataset(image):
    if image=='test':
        data = df[df['test']==1]
        #data = df[df['trained']==1]
    elif image=='random':
        data = df[df['trained']==0].sample(100).reset_index()
        #data = df[df['test']==0].sample(100).reset_index()
    else:
        data = df[df['Image Index']==image]
    return data.reset_index()

#build image dataset according to data
def buildImageset(df):
    start=time.time()
    sample_size = df['Image Index'].count()
    Y = np.ndarray((sample_size,1), dtype=np.uint8)
    X = np.ndarray((sample_size, img_width, img_height, 3), dtype=np.float32)
    #i = 0

    #import images as array:
    for index, row in df.iterrows():
        if row['Augmented']==1:
            d = DIRECT_AUGMENTED
        else:
            d = DIRECT_ORIGINALS
        # Load image in grayscale
        img = cv2.imread(d+row['Image Index'],1)
        #resize
        img = img[crop_x:crop_x+crop_w,crop_y:crop_y+crop_h]
        img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_AREA)
        #img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = img_to_array(img)

        X[index] = img.astype('float32') / 255
        #X[index] = (img - img.min())/(img.max() - img.min())
        Y[index] = row[PATHOLOGY_NAME]

    start1=time.time()
    print("loop build images set duration :" +str(start1-start)+"sec")
    print("X : "+str(X.shape))
    print("Y : "+str(Y.shape))
    
    return X, Y


#Load model
model = loadModel()
dataTest = loadDataset(IMAGE_NAME)

X_test, y_test = buildImageset(dataTest)
x_test_mean = np.mean(X_test, axis=0)
X_test -= x_test_mean
print("expected results :\n"+str(y_test))
#y_test = to_categorical(y_test, num_classes=2)
#print("categorical expected results :\n"+str(y_test))
if IMAGE_NAME=='test' or IMAGE_NAME=='random':
    score = model.evaluate(X_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    out = model.predict(X_test)
    print('predict: '+str(out))

    out = np.array(out)

    threshold = np.arange(0.01,0.99,0.01)

    np.seterr(divide='ignore', invalid='ignore')

    acc = []
    accuracies = []
    best_threshold = np.zeros(out.shape[1])
    for i in range(out.shape[1]):
        y_prob = np.array(out[:,i])
        for j in threshold:
            y_pred = [1 if prob>=j else 0 for prob in y_prob]
            acc.append( matthews_corrcoef(y_test[:,i],y_pred))
        acc   = np.array(acc)
        index = np.where(acc==acc.max()) 
        accuracies.append(acc.max()) 
        best_threshold[i] = threshold[index[0][0]]
        acc = []

    print("best_threshold : "+str(best_threshold))
    y_pred = np.array([[1 if out[i,j]>=best_threshold[j] else 0 for j in range(y_test.shape[1])] for i in range(len(y_test))])
    
    print("hamming loss : "+str(hamming_loss(y_test,y_pred)))  #the loss should be as low as possible and the range is from 0 to 1
    print("results :\n"+str(y_pred))
    total_correctly_predicted = len([i for i in range(len(y_test)) if (y_test[i]==y_pred[i]).sum() == 1])
    print("totel correct : "+str(total_correctly_predicted))

    print("ratio correct predict: "+str(total_correctly_predicted/float(len(y_test))))
else:
    #Check model on image dataset
    scores = model.predict(X_test)
    print("result before threshold :"+str(scores))
    #best_threshold=[0.007]
    #y_pred = np.array([1 if scores[0,i]>=best_threshold[i] else 0 for i in range(scores.shape[1])])
    #print("results :\n"+str(y_pred))
