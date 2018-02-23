# coding: utf-8
import numpy as np
import pandas as pd
import os
import cv2
import time
import keras
import sys
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import hamming_loss
from sklearn.metrics import precision_score, recall_score, f1_score
import tensorflow as tf
import keras.backend.tensorflow_backend as tfb

DIRECT_ORIGINALS='../images/'
DIRECT_AUGMENTED='../imagesAugmented/'
DEFAULT_MODEL_NAME='myModel.h5'
NUMBER_OF_DESEASES=14
POS_WEIGHT = 20  # multiplier for positive targets, needs to be tuned

img_width, img_height = 512, 512
input_shape = (img_width, img_height,1)

batch_size=4
#cropping dimension
crop_x,crop_y,crop_w,crop_h=(112,112,800,800)
np.random.seed(1234)

#columns for each desease
pathology_list = ['Cardiomegaly','Emphysema','Effusion','Hernia','Nodule','Pneumothorax','Atelectasis','Pleural_Thickening','Mass','Edema','Consolidation','Infiltration','Fibrosis','Pneumonia','No Finding']

#import image data set description
df = pd.read_csv('Data_augmented.csv')


def weighted_binary_crossentropy(target, output):
    """
    Weighted binary crossentropy between an output tensor 
    and a target tensor. POS_WEIGHT is used as a multiplier 
    for the positive targets.

    Combination of the following functions:
    * keras.losses.binary_crossentropy
    * keras.backend.tensorflow_backend.binary_crossentropy
    * tf.nn.weighted_cross_entropy_with_logits
    """
    # transform back to logits
    _epsilon = tfb._to_tensor(tfb.epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
    output = tf.log(output / (1 - output))
    # compute weighted loss
    loss = tf.nn.weighted_cross_entropy_with_logits(targets=target,
                                                    logits=output,
                                                    pos_weight=POS_WEIGHT)
    return tf.reduce_mean(loss, axis=-1)


#loadModel : create or load already trained model
def loadModel():
    print("loadModel : "+MODEL_NAME)
    if os.path.exists(MODEL_NAME):
        mod = load_model(MODEL_NAME,custom_objects={'weighted_binary_crossentropy': weighted_binary_crossentropy})
    else:
        print("ERROR: model "+MODEL_NAME+" doesnt exist !!")
        sys.exit()
    return mod


#Create dataset for validation
def loadDataset(image):
    if image=='test':
        data = df[df['test']==1].reset_index()
    elif image=='random':
        data = df[df['trained']==0].sample(1000).reset_index()
    else:
        data = df[df['Image Index']==image].reset_index()
    return data

#build image dataset according to data
def buildImageset(df):
    start=time.time()
    sample_size = df['Image Index'].count()
    Y = np.ndarray((sample_size, NUMBER_OF_DESEASES), dtype=np.float32)
    X = np.ndarray((sample_size, img_width, img_height, 1), dtype=np.float32)

    pat_list = ['Cardiomegaly','Emphysema','Effusion','Hernia','Nodule','Pneumothorax','Atelectasis','Pleural_Thickening','Mass','Edema','Consolidation','Infiltration','Fibrosis','Pneumonia']

    #import images as array
    for index, row in df.iterrows():
        d = DIRECT_ORIGINALS
        # Load image in grayscale
        img = cv2.imread(d+row['Image Index'],0)
        #resize
        img = img[crop_x:crop_x+crop_w,crop_y:crop_y+crop_h]
        img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_AREA)
        img = img_to_array(img)
        X[index] = (img - img.min())/(img.max() - img.min())
        Y[index] = row[pat_list]

    start1=time.time()
    print("loop build images set duration :" +str(start1-start)+"sec")
    print("X : "+str(X.shape))
    print("Y : "+str(Y.shape))
    
    return X, Y


#load and build training set
IMAGE_NAME = raw_input("Please enter image name or 'random' for small random validation or 'test' for full validation: ")
if len(IMAGE_NAME)==0:
    IMAGE_NAME='test'

MODEL_NAME = raw_input("Please enter model name : ")
if len(MODEL_NAME)==0:
    MODEL_NAME = DEFAULT_MODEL_NAME

#Load model
model = loadModel()

dataTest = loadDataset(IMAGE_NAME)
X_test, y_test = buildImageset(dataTest)
#print("expected results :\n"+str(y_test))

if IMAGE_NAME=='test' or IMAGE_NAME=='random':
    score = model.evaluate(X_test, y_test, verbose=1, batch_size=batch_size)
    print("Test loss: "+str(score[0]))
    print("Test accuracy: "+str(score[1]))
    out = model.predict(X_test, batch_size=batch_size)
    #print('predict: '+str(out))
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

    print("best_threshold : "+str(best_threshold)+"\n")
    for j in range(y_test.shape[1]):
        y_pred = np.array([1 if out[i,j]>=best_threshold[j] else 0 for i in range(len(y_test))])
        print("\nResults for "+str(pathology_list[j]) +" :")
        print("   Test size : "+str(len(y_test)))
        positive = np.array([1 if y_test[i,j]==1 else 0 for i in range(len(y_test))]).sum()
        print("   Total number of desease : "+str(positive))
        positive_predict = np.array([1 if y_pred[i]==1 else 0 for i in range(len(y_test))]).sum()
        print("   Total number of predict desease : "+str(positive_predict))
        positive_match = np.array([1 if (y_test[i,j]==1 and y_pred[i]==1) else 0 for i in range(len(y_test))]).sum()
        print("   Total number of matching desease : "+str(positive_match)+"  Ratio : "+str(positive_match/(1.*positive)))
        negative = np.array([1 if y_test[i,j]==0 else 0 for i in range(len(y_test))]).sum()
        negative_match = np.array([1 if (y_test[i,j]==0 and y_pred[i]==0) else 0 for i in range(len(y_test))]).sum()
        print("   Total number of matching no desease : "+str(negative_match)+"  Ratio : "+str(negative_match/(1.*negative)))
        total_correctly_predicted = len([i for i in range(len(y_test)) if (y_test[i,j]==y_pred[i]).sum() == 1])
        print("   Totel correct : "+str(total_correctly_predicted)+"  Ratio : "+str(total_correctly_predicted/(1.*len(y_test))))
        print("Other metrics :")
        print("   Hamming loss : "+str(hamming_loss(y_test[:,j],y_pred)))
        precision = precision_score(y_test[:,j], y_pred, average='weighted')
        recall = recall_score(y_test[:,j], y_pred, average='weighted')
        f1 = f1_score(y_test[:,j], y_pred, average="weighted")
        print("   Precision: "+str(precision))
        print("   Recall: "+str(recall))
        print("   F1: "+str(f1))

    #y_pred = np.array([[1 if out[i,j]>=best_threshold[j] else 0 for j in range(y_test.shape[1])] for i in range(len(y_test))])

    #print("hamming loss : "+str(hamming_loss(y_test,y_pred)))  #the loss should be as low as possible and the range is from 0 to 1
    #print("results :\n"+str(y_pred))
    #total_correctly_predicted = len([i for i in range(len(y_test)) if (y_test[i]==y_pred[i]).sum() == NUMBER_OF_DESEASES])
    #print("totel correct : "+str(total_correctly_predicted))

    #print("ratio correct predict: "+str(total_correctly_predicted/float(len(y_test))))

else:
    #Check model on Test dataset
    scores = model.predict(X_test)
    best_threshold=[ 0.46, 0.44, 0.46, 0.65, 0.47, 0.06, 0.45, 0.46, 0.92, 0.47, 0.52, 0.24, 0.58, 0.56]
    print(scores)
    y_pred = np.array([1 if scores[0,i]>=best_threshold[i] else 0 for i in range(scores.shape[1])])
    print("results :\n"+str(y_pred))

