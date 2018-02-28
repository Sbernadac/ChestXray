# coding: utf-8
import numpy as np
import pandas as pd
import os
import cv2
import time
import keras
import matplotlib.pyplot as plt
import sys, getopt
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.models import Sequential, load_model, Model
from keras.utils import to_categorical
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import hamming_loss
from sklearn.metrics import precision_score, recall_score, f1_score


##########################################
############# DEFINE #####################
DIRECT_ORIGINALS='../../images/'
DIRECT_AUGMENTED='../../imagesAugmented/'


batch_size=4
NUMBER_OF_DESEASES=1

#cropping dimension
crop_x,crop_y,crop_w,crop_h=(112,112,800,800)
np.random.seed(1234)


#loadModel : create or load already trained model
def loadModel(MODEL_NAME):
    if os.path.exists(MODEL_NAME):
        model = load_model(MODEL_NAME)
    else:
        sys.exit(1)
    return model


#Create dataset for validation
def loadDataset(image,file):
    #import image data set description
    df = pd.read_csv(file)

    if image=='test':
        data = df[df['test']==1].sample(2000)
    elif image=='random':
        data = df[df['trained']==0].sample(100).reset_index()
        #data = df[df['test']==0].sample(100).reset_index():60
    else:
        data = df[df['Image Index']==image]
    return data.reset_index()

#build image dataset according to data
def buildImageset(df,PATHOLOGY_NAME,img_width, img_height):
    start=time.time()
    sample_size = df['Image Index'].count()
    print("buildImageset size : "+str(sample_size))
    Y = np.ndarray((sample_size,1), dtype=np.float32)
    X = np.ndarray((sample_size, img_width, img_height, 1), dtype=np.float32)

    #import images as array
    for index, row in df.iterrows():
        if row['Augmented']==1:
            d = DIRECT_AUGMENTED
        else:
            d = DIRECT_ORIGINALS
        # Load image in grayscale
        img = cv2.imread(d+row['Image Index'],0)
        #crop image
        img = img[crop_x:crop_x+crop_w,crop_y:crop_y+crop_h]
        img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_AREA)
        #img = img.transpose()/255.0
        img = img_to_array(img)
        X[index] = (img - img.min())/(img.max() - img.min())
        Y[index] = row[PATHOLOGY_NAME]
    
    end=time.time()
    print("loop build images set duration :" +str(end-start)+"sec")
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

################################################
############### MAIN ###########################
################################################

def main(argv):

    #Defaults values
    PATHOLOGY_NAME='Cardiomegaly'
    MODEL_NAME='myCardiomegaly2000.h5'
    IMAGE_NAME='test'
    shape = 224

    try:
        opts, args = getopt.getopt(argv,"hp:m:t:s:",["pathology=","model=","test=","shape="])
    except getopt.GetoptError:
        print 'test.py -p <pathology> -m <model> -t <test> -s <shape>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'test.py -p <pathology> -m <model> -t <test> -s <shape>'
            sys.exit()
        elif opt in ("-p", "--pathology"):
            PATHOLOGY_NAME = arg
        elif opt in ("-m", "--model"):
            MODEL_NAME = arg
	elif opt in ("-t", "--test"):
            IMAGE_NAME = arg
        elif opt in ("-s", "--shape"):
            shape = int(arg)
    print 'Pathology is ', PATHOLOGY_NAME
    print 'Pathology model is ', MODEL_NAME
    print 'Test is ', IMAGE_NAME
    print 'Shape is ', str(shape)

    FILE_NAME="Data_"+PATHOLOGY_NAME+".csv"
    img_width, img_height = shape, shape
    
    model = loadModel(MODEL_NAME)
    dataTest = loadDataset(IMAGE_NAME,FILE_NAME)

    X_test, y_test = buildImageset(dataTest,PATHOLOGY_NAME,img_width, img_height)
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
        print("total correct : "+str(total_correctly_predicted))

        print("ratio correct predict: "+str(total_correctly_predicted/float(len(y_test))))
    
        true_positive = np.array([1 if (y_test[i]==1) else 0 for i in range(len(y_test))]).sum()
        false_positive = np.array([1 if (y_test[i]==0 and y_pred[i]==1) else 0 for i in range(len(y_test))]).sum()
        correct_positive = np.array([1 if (y_test[i]==1 and y_pred[i]==1) else 0 for i in range(len(y_test))]).sum()
        print("Positive: false="+str(false_positive)+ ", true="+str(true_positive)+ ", correct="+str(correct_positive)+ ", ratio="+str(correct_positive/(true_positive*1.0)))
        true_negative = np.array([1 if (y_test[i]==0) else 0 for i in range(len(y_test))]).sum()
        false_negative = np.array([1 if (y_test[i]==1 and y_pred[i]==0) else 0 for i in range(len(y_test))]).sum()
        correct_negative = np.array([1 if (y_test[i]==0 and y_pred[i]==0) else 0 for i in range(len(y_test))]).sum()
        print("Negative: false="+str(false_negative)+ ", true="+str(true_negative)+ ", correct="+str(correct_negative)+ ", ratio="+str(correct_negative/(true_negative*1.0)))

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
        
        
if __name__ == "__main__":
    main(sys.argv[1:])
