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
from sklearn.metrics import precision_score, recall_score, f1_score


DIRECT_ORIGINALS='../../images/'
DIRECT_AUGMENTED='../../imagesAugmented/'


batch_size=4

img_width, img_height = 256, 256
NUMBER_OF_DESEASES=1
input_shape = (img_width, img_height,NUMBER_OF_DESEASES)
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
        df['current_test'] = 0
        #extract current pathology
        df.loc[(df[PATHOLOGY_NAME]==1)&(df['test']==1),['current_test']]=1
        size = df[df['current_test']==1]['Image Index'].count()
        print("Current pathology test size : "+str(size))
        df.loc[df[(df[PATHOLOGY_NAME]==0)&(df['test']==1)].sample(size).index,['current_test']]=1
        data = df[df['current_test']==1]
    elif image=='random':
        data = df[df['trained']==0].sample(100).reset_index()
        #data = df[df['test']==0].sample(100).reset_index():60
    else:
        data = df[df['Image Index']==image]
    return data.reset_index()

#build image dataset according to data
def buildImageset(df):
    start=time.time()
    sample_size = df['Image Index'].count()
    Y = np.ndarray((sample_size,1), dtype=np.uint8)
    X = np.ndarray((sample_size, img_width, img_height, 1), dtype=np.float32)
    #i = 0

    #import images as array:
    for index, row in df.iterrows():
        if row['Augmented']==1:
            d = DIRECT_AUGMENTED
        else:
            d = DIRECT_ORIGINALS
        # Load image in grayscale
        img = cv2.imread(d+row['Image Index'],0)
        #resize
        img = img[crop_x:crop_x+crop_w,crop_y:crop_y+crop_h]
        img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_AREA)
        #img = img.transpose()/255.0
        img = img_to_array(img)
        X[index] = (img - img.min())/(img.max() - img.min())
        Y[index] = row[PATHOLOGY_NAME]
        #X[i] = (img - img.min())/(img.max() - img.min())
        #Y[i]=data.iloc[i][PATHOLOGY_NAME]
        #i += 1

    start1=time.time()
    print("loop build images set duration :" +str(start1-start)+"sec")
    print("X : "+str(X.shape))
    print("Y : "+str(Y.shape))
    
    return X, Y

#compute best threshold
def bestthreshold(threshold,out,y_test):
    acc = []
    accuracies = []
    best_threshold = 0
    y_prob = np.array(out[:,0])
    for j in threshold:
        y_pred = [1 if prob>=j else 0 for prob in y_prob]
        acc.append( matthews_corrcoef(y_test[:,0],y_pred))
    acc   = np.array(acc)
    index = np.where(acc==acc.max())
    accuracies.append(acc.max())
    best_threshold = threshold[index[0][0]]

    return best_threshold


#Load model
model = loadModel()
dataTest = loadDataset(IMAGE_NAME)

X_test, y_test = buildImageset(dataTest)
#y_test = to_categorical(y_test, num_classes=2)
#print("categorical expected results :\n"+str(y_test))
if IMAGE_NAME=='test' or IMAGE_NAME=='random':
    score = model.evaluate(X_test, y_test, verbose=1, batch_size=batch_size)
    print("\n\n\n###########################")
    print("######## RESULTS ##########\n")
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    out = model.predict(X_test, batch_size=batch_size)

    out = np.array(out)

    np.seterr(divide='ignore', invalid='ignore')

    threshold = np.arange(0.01,0.99,0.01)
    best_threshold = bestthreshold(threshold,out,y_test)
    print("best_threshold : "+str(best_threshold))
    if best_threshold<0.02:
        #try to find a better one
        threshold = np.arange(0.001,0.01,0.001)
        best_threshold = bestthreshold(threshold,out,y_test)
        print("best_threshold : "+str(best_threshold))

    y_pred = np.array([1 if out[i,0]>=best_threshold else 0 for i in range(len(y_test))])
    
    print("hamming loss : "+str(hamming_loss(y_test,y_pred)))  #the loss should be as low as possible and the range is from 0 to 1
    #print("results :\n"+str(y_pred))
    total_correctly_predicted = len([i for i in range(len(y_test)) if (y_test[i]==y_pred[i]).sum() == 1])
    print("totel correct : "+str(total_correctly_predicted))

    print("ratio correct predict: "+str(total_correctly_predicted/float(len(y_test))))
    
    false_positive = np.array([1 if (y_test[i]==0 and y_pred[i]==1) else 0 for i in range(len(y_test))]).sum()
    print("false_positive = "+str(false_positive))
    false_negative = np.array([1 if (y_test[i]==1 and y_pred[i]==0) else 0 for i in range(len(y_test))]).sum()
    print("false_negative = "+str(false_negative))

    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average="weighted")
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)
else:
    #Check model on image dataset
    scores = model.predict(X_test)
    print("result before threshold :"+str(scores))
    #best_threshold=[0.007]
    #y_pred = np.array([1 if scores[0,i]>=best_threshold[i] else 0 for i in range(scores.shape[1])])
    #print("results :\n"+str(y_pred))
