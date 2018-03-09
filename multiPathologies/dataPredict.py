# coding: utf-8
import numpy as np
import pandas as pd
import os
import cv2
import time
import keras
import sys, getopt
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import hamming_loss
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import tensorflow as tf
import keras.backend.tensorflow_backend as tfb


##########################################
############# DEFINE #####################
DIRECT_ORIGINALS='../images/'
DIRECT_AUGMENTED='../imagesAugmented/'
NUMBER_OF_DESEASES=14
POS_WEIGHT = 20  # multiplier for positive targets, needs to be tuned


batch_size=4
#cropping dimension
crop_x,crop_y,crop_w,crop_h=(112,112,800,800)
np.random.seed(1234)


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
def loadModel(MODEL_NAME):
    if os.path.exists(MODEL_NAME):
        mod = load_model(MODEL_NAME,custom_objects={'weighted_binary_crossentropy': weighted_binary_crossentropy})
    else:
        print("ERROR: model "+MODEL_NAME+" doesnt exist !!")
        sys.exit()
    return mod


#Create dataset for validation
def loadDataset(image):
    #import image data set description
    df = pd.read_csv('Data_augmented.csv')

    if image=='test':
        data = df[df['test']==1]
    elif image=='random':
        data = df[df['test']==1].sample(2000)
    else:
        data = df[df['Image Index']==image]
    return data.reset_index()

#build image dataset according to data
def buildImageset(df,img_width, img_height, third_dim):
    start=time.time()
    sample_size = df['Image Index'].count()
    Y = np.ndarray((sample_size, NUMBER_OF_DESEASES), dtype=np.float32)
    X = np.ndarray((sample_size, img_width, img_height, third_dim), dtype=np.float32)

    pat_list = ['Cardiomegaly','Emphysema','Effusion','Hernia','Nodule','Pneumothorax','Atelectasis','Pleural_Thickening','Mass','Edema','Consolidation','Infiltration','Fibrosis','Pneumonia']

    #import images as array
    for index, row in df.iterrows():
        d = DIRECT_ORIGINALS
        # Load image in grayscale
        img = cv2.imread(d+row['Image Index'],0)
        #crop image
        img = img[crop_x:crop_x+crop_w,crop_y:crop_y+crop_h]
        img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_AREA)
        img = img_to_array(img)
        X[index] = (img - img.min())/(img.max() - img.min())
        Y[index] = row[pat_list]

    end=time.time()
    print("loop build images set duration :" +str(end-start)+"sec")
    print("X : "+str(X.shape))
    print("Y : "+str(Y.shape))
    
    return X, Y

#compute best threshold
def bestthreshold(threshold,out,y_test):
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
    return best_threshold

################################################
############### MAIN ###########################
################################################

def main(argv):

    #Defaults values
    MODEL_NAME='myModel.h5'
    IMAGE_NAME='test'
    shape = 224
    third_dim = 1


    try:
        opts, args = getopt.getopt(argv,"hm:t:s:d:",["model=","test=","shape=","third_dim="])
    except getopt.GetoptError:
        print 'dataPredict.py  -m <model> -t <test> -s <shape> -d <third_dim>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'dataPredict.py -m <model> -t <test> -s <shape> -d <third_dim>'
            sys.exit()
        elif opt in ("-m", "--model"):
            MODEL_NAME = arg
	elif opt in ("-t", "--test"):
            IMAGE_NAME = arg
        elif opt in ("-s", "--shape"):
            shape = int(arg)
        elif opt in ("-d", "--third_dim"):
            third_dim = int(arg)
    print 'Pathology model is ', MODEL_NAME
    print 'Test is ', IMAGE_NAME
    print 'Shape is ', str(shape)
    print 'Third dimension is ', str(third_dim)

    img_width, img_height = shape, shape
    
    model = loadModel(MODEL_NAME)
    dataTest = loadDataset(IMAGE_NAME)

    #columns for each desease
    pathology_list = ['Cardiomegaly','Emphysema','Effusion','Hernia','Nodule','Pneumothorax','Atelectasis','Pleural_Thickening','Mass','Edema','Consolidation','Infiltration','Fibrosis','Pneumonia','No Finding']


    X_test, y_test = buildImageset(dataTest,img_width, img_height, third_dim)
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
        for i in range(len(best_threshold)):
            if best_threshold[i]<0.02:
                #try to find a better one
                threshold = np.arange(0.001,0.01,0.001)
                acc = []
                accuracies = []
                y_prob = np.array(out[:,i])
                for j in threshold:
                    y_pred = [1 if prob>=j else 0 for prob in y_prob]
                    acc.append( matthews_corrcoef(y_test[:,i],y_pred))
                acc   = np.array(acc)
                index = np.where(acc==acc.max())
                accuracies.append(acc.max())
                best_threshold[i] = threshold[index[0][0]]

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
            negative_false = np.array([1 if (y_test[i,j]==1 and y_pred[i]==0) else 0 for i in range(len(y_test))]).sum()
            positive_false = np.array([1 if (y_test[i,j]==0 and y_pred[i]==1) else 0 for i in range(len(y_test))]).sum()
            print("   Total number of False negative : "+str(negative_false)+"  False positive : "+str(positive_false))
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
            roc = roc_auc_score(y_test[:,j], y_pred)
            print("   Precision: "+str(precision))
            print("   Recall: "+str(recall))
            print("   F1: "+str(f1))
            print("   AUC: "+str(roc))


        y_pred = np.array([[1 if out[i,j]>=best_threshold[j] else 0 for j in range(y_test.shape[1])] for i in range(len(y_test))])
        total_correctly_predicted = len([i for i in range(len(y_test)) if (y_test[i]==y_pred[i]).sum() == NUMBER_OF_DESEASES])
        print("\nGlobal totel correct : "+str(total_correctly_predicted))
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average="weighted")
        roc = roc_auc_score(y_test, y_pred)
        print("Global Precision: "+str(precision))
        print("Global Recall: "+str(recall))
        print("Global F1: "+str(f1))
        print("Global AUC: "+str(roc))


    else:
        #Check model on image dataset
        scores = model.predict(X_test)
        print("result before threshold :"+str(scores))

        
        
if __name__ == "__main__":
    main(sys.argv[1:])
