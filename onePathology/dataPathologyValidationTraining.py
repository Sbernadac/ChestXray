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
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import SGD,Adagrad,Adadelta,RMSprop,Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from keras.regularizers import l2, l1, l1_l2
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import hamming_loss
from sklearn.metrics import precision_score, recall_score, f1_score


##########################################
############# DEFINE #####################
DIRECT_ORIGINALS='../../images/'
DIRECT_AUGMENTED='../../imagesAugmented/'
AUGMENTED_FILE='Data_augmented.csv'
OUTPUT_DIR="output/"


batch_size=8
epoch=50
NUMBER_OF_DESEASES=1
img_width, img_height = 512 ,512
input_shape = (img_width, img_height,1)
#cropping dimension
crop_x,crop_y,crop_w,crop_h=(112,112,800,800)

np.random.seed(1234)

##########################################################
############### Functions ################################
##########################################################
def hybrid_pool_layer(x, pool_size=(2,2)):
    return Conv2D(int(x.shape[-1]), (1, 1))(
        keras.layers.concatenate([
            MaxPooling2D(pool_size)(x),
            AveragePooling2D(pool_size)(x)]))

def createDataSet(FILE_NAME):
    #check if training DataFrame already exists
    if os.path.exists(FILE_NAME):
        data = pd.read_csv(FILE_NAME)
    #create new training DataFrame
    else:
        #import original image data set description
        data = pd.read_csv(AUGMENTED_FILE)

        #Add column for already trained:
        data['trained']=0

        #Add column for validation:
        data['validation']=0

    print("Training and validation dataset size : "+str(data[data['test']==0]['Image Index'].count()))
    #print("Test dataset size : "+str(data[data['test']==1]['Image Index'].count()))
    return data

#loadModel : create or load already trained model
def loadModel(MODEL_NAME):
    if os.path.exists(MODEL_NAME):
        model = load_model(MODEL_NAME)
    else:
        img_input = Input(shape=input_shape)
        #kernel_regularizer=l1_l2(l1=0.01, l2=0.01)
        x = Conv2D(64, kernel_size=11, padding='same', strides=2, activity_regularizer=l2(0.001))(img_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = AveragePooling2D(pool_size=(2,2))(x)
        #x = hybrid_pool_layer(x,pool_size=(3,3))
        #x = Dropout(0.2)(x)

        x = Conv2D(96, kernel_size=7, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = AveragePooling2D(pool_size=(2,2))(x)

        x = Conv2D(128, kernel_size=7, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = AveragePooling2D(pool_size=(2,2))(x)
        #x = hybrid_pool_layer(x,pool_size=(3,3))
        #x = Dropout(0.2)(x)

        x = Conv2D(160, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = AveragePooling2D(pool_size=(2,2))(x)

        x = Conv2D(192, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = AveragePooling2D(pool_size=(2,2))(x)
        #x = hybrid_pool_layer(x,pool_size=(2,2))
        x = Dropout(0.2)(x)

        x = Conv2D(256, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(256, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = AveragePooling2D(pool_size=(2,2))(x)
        #x = hybrid_pool_layer(x,pool_size=(2,2))
        x = Dropout(0.2)(x)

        x = Conv2D(384, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(384, kernel_size=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(512, kernel_size=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = AveragePooling2D(pool_size=(3,3))(x)
        #x = hybrid_pool_layer(x,pool_size=(3,3))

        x = Flatten()(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(1, activation='sigmoid')(x)
        model = Model(img_input, x)
    

    print(model.summary())
    return model

#Create datasets
def loadTrainingDataset(df,PATHOLOGY_NAME,DATASET_SIZE):
    #build dataset to train
    df['currentTraining']=0

    total_size = df[(df[PATHOLOGY_NAME]==1)&(df['test']==0)&(df['trained']==0)][PATHOLOGY_NAME].count()
    total_size = min(DATASET_SIZE,total_size)
    validation_size = total_size/10
    total_size = total_size - validation_size
    df.loc[df[(df[PATHOLOGY_NAME]==1)&(df['test']==0)&(df['trained']==0)].sample(total_size).index,['currentTraining']]=1
    print(PATHOLOGY_NAME+" size : "+str(total_size))
    #add no pathology
    df.loc[df[(df[PATHOLOGY_NAME]==0)&(df['test']==0)&(df['trained']==0)].sample(total_size).index,['currentTraining']]=1

    #create validation set
    df.loc[df[(df[PATHOLOGY_NAME]==1)&(df['test']==0)&(df['trained']==0)&(df['currentTraining']==0)].sample(validation_size).index,['validation']]=1
    #add no pathology
    df.loc[df[(df[PATHOLOGY_NAME]==0)&(df['test']==0)&(df['trained']==0)&(df['currentTraining']==0)].sample(validation_size).index,['validation']]=1

    return 

#Create dataset for validation
def loadDataset(df,PATHOLOGY_NAME):
    #extract validation
    data = df[df['validation']==1]
    return data.reset_index()

#build image dataset according to data
def buildImageset(df,PATHOLOGY_NAME):
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
        img = img_to_array(img)
        X[index] = (img - img.min())/(img.max() - img.min())
        Y[index] = row[PATHOLOGY_NAME]
    
    end=time.time()
    print("loop build images set duration :" +str(end-start)+"sec")
    print("X : "+str(X.shape))
    print("Y : "+str(Y.shape))
    return X, Y


def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    #lr = 1e-3
    lr = 1e-4
    if epoch > 120:
        lr *= 0.5e-3
    elif epoch > 100:
        lr *= 1e-3
    elif epoch > 80:
        lr *= 1e-2
    elif epoch > 40:
        lr *= 1e-1
    print("Learning rate: "+ str(lr))
    return lr


def history_plot(history,PICTURE_NAME):
    plt.figure(1)

    plt.subplot(211)
    plt.plot(history.history['acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')

    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')

    plt.savefig(OUTPUT_DIR+PICTURE_NAME)

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
    DATASET_SIZE=2000

    try:
        opts, args = getopt.getopt(argv,"hp:s:",["pathology=","size="])
    except getopt.GetoptError:
        print 'test.py -p <pathology> -s <size>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'test.py -p <pathology> -s <size>'
            sys.exit()
        elif opt in ("-p", "--pathology"):
            PATHOLOGY_NAME = arg
        elif opt in ("-s", "--size"):
            DATASET_SIZE = int(arg)
    print 'Pathology is ', PATHOLOGY_NAME
    print 'Dataset pathology size is ', DATASET_SIZE

    MODEL_NAME="my"+PATHOLOGY_NAME+str(DATASET_SIZE)+".h5"
    FILE_NAME="Data_"+PATHOLOGY_NAME+str(DATASET_SIZE)+".csv"
    PICTURE_NAME=PATHOLOGY_NAME+str(DATASET_SIZE)+".png"


    #Load model
    model = loadModel(MODEL_NAME)
    # SGD > RMSprop > Adam
    #sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    opt = Adam(lr=lr_schedule(0))
    #optimizer = RMSprop(lr=lr_schedule(0), decay=1e-6)
    model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

    #load and build training set
    data = createDataSet(FILE_NAME)
    df = loadTrainingDataset(data,PATHOLOGY_NAME,DATASET_SIZE)
    dataTrain = df[df['currentTraining']==1].reset_index()
    X_train, Y_train = buildImageset(dataTrain,PATHOLOGY_NAME)


    #Add callback to monitor model quality
    filepath=OUTPUT_DIR+MODEL_NAME+"-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)

    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)
    #callbacks_list = [checkpoint,lr_scheduler,lr_reducer]
    callbacks_list = [lr_scheduler]

    #train model
    #Y_train = to_categorical(Y_train, num_classes=2)
    start=time.time()
    history=model.fit(X_train, Y_train, batch_size=batch_size, epochs=epoch, shuffle=True, callbacks=callbacks_list, validation_split=0., verbose=1)
    end=time.time()
    print("fit duration :" +str(end-start)+"sec")

    #save model
    model.save(OUTPUT_DIR+MODEL_NAME)

    #set Trained to 1
    data.loc[data['currentTraining']==1,['trained']]=1
    print("Already trained images : "+str(data[data['trained']==1]['Image Index'].count()))
    #save current step to training file 
    data.to_csv(OUTPUT_DIR+FILE_NAME,index=False)

    #plot history
    history_plot(history,PICTURE_NAME)


    #Check Results
    dataTest = loadDataset(data,PATHOLOGY_NAME)
    X_test, y_test = buildImageset(dataTest,PATHOLOGY_NAME)
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


if __name__ == "__main__":
    main(sys.argv[1:])