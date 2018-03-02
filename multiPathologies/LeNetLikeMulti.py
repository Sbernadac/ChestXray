# coding: utf-8
import numpy as np
import pandas as pd
import os
import cv2
import time
import keras
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.models import Sequential, load_model, Model
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import SGD,Adagrad,Adadelta,RMSprop,Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from keras.regularizers import l2, l1, l1_l2
import tensorflow as tf
import keras.backend.tensorflow_backend as tfb

##########################################
############# DEFINE #####################
DIRECT_ORIGINALS='../images/'
DIRECT_AUGMENTED='../imagesAugmented/'
AUGMENTED_FILE='Data_augmented.csv'
SAVED_FILE='Data_Training_steps.csv'
MODEL_NAME="myModel.h5"

NUMBER_OF_DESEASES=14
DESEASES_TRAINING_SIZE=1000
POS_WEIGHT = 20  # multiplier for positive targets, needs to be tuned

batch_size=4
epoch=100
img_width, img_height = 224 ,224
input_shape = (img_width, img_height,1)
#cropping dimension
crop_x,crop_y,crop_w,crop_h=(112,112,800,800)

np.random.seed(1234)


#columns for each desease
pathology_list = ['Hernia','Pneumonia','Fibrosis','Edema','Emphysema','Cardiomegaly','Pleural_Thickening','Consolidation','Pneumothorax','Mass','Nodule','Effusion','Atelectasis','Infiltration','No Finding']


##########################################################
############### Functions ################################
##########################################################
def createDataSet():
    #check if training DataFrame already exists
    if os.path.exists(SAVED_FILE):
        data = pd.read_csv(SAVED_FILE)
    #create new training DataFrame
    else:
        #import image data set description
        data = pd.read_csv(AUGMENTED_FILE)

        #Add column for already trained:
        data['trained']=0

    print("Training and validation dataset size : "+str(data[data['test']==0]['Image Index'].count()))
    print("Test dataset size : "+str(data[data['test']==1]['Image Index'].count()))
    return data


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


#define loss metric
def rloss(y_true, y_pred):
    g = tf.subtract(tf.expand_dims(y_pred[1], -1), y_pred[1])
    g = tf.cast(g == 0.0, tf.float32) * 0.5 + tf.cast(g > 0.0, tf.float32)
    f = tf.subtract(tf.expand_dims(y_true[1], -1), y_true[1]) > 0.0
    f = tf.matrix_band_part(tf.cast(f, tf.float32), -1, 0)
    g = tf.reduce_sum(tf.multiply(g, f))
    f = tf.reduce_sum(f)
    return tf.where(tf.equal(g, 0), 0.0, g/f)


#loadModel : create or load already trained model
def loadModel():
    if os.path.exists(MODEL_NAME):
        print("loadModel : "+MODEL_NAME+" already exists")
        model = load_model(MODEL_NAME)
    else:
        model = Sequential()
        model.add(Conv2D(20, kernel_size=5, padding='same', strides=2, input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.3))

        model.add(Conv2D(50, kernel_size=5, padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.3))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(500))
        model.add(Activation('relu'))
        model.add(Dropout(0.4))
        model.add(Dense(NUMBER_OF_DESEASES))
        model.add(Activation('sigmoid'))
    

    print(model.summary())
    #print(model.get_config())
    return model

#Create dataset for training
def loadTrainingDataset(df):
    #build dataset to train
    df['currentTraining']=0

    #check if we need to slip dataset
    untrained_size=df[(df['test']==0)&(df['trained']==0)]['Image Index'].count()
    print("Untrained size : "+str(untrained_size))

    if untrained_size > DESEASES_TRAINING_SIZE*NUMBER_OF_DESEASES:
        deseases={}
        #update sample fraction accordingly to untrained size
        for pathology in pathology_list :
	    deseases[pathology]=df[(df[pathology]==1)&(df['trained']==0)&(df['test']==0)&(df['currentTraining']==0)][pathology].count()
        #sort deseases
        sort_deseases=sorted(deseases.items(), key=lambda x: x[1])

        for pat,num in sort_deseases:
            #select desease sample, need to take care of multiple deseases
            count = min (DESEASES_TRAINING_SIZE,num)  #take the min between wanted and available
            count = count - df[(df['currentTraining']==1)&(df[pat]==1)][pat].count() #remove images already selected for training
            if count > 0:
                print("take "+str(count)+" for "+pat)
	        df.loc[df[(df[pat]==1)&(df['test']==0)&(df['trained']==0)&(df['currentTraining']==0)].sample(count).index,['currentTraining']]=1
        
    else:
        df.loc[df[(df['test']==0)&(df['trained']==0)],['currentTraining']]=1
    print("Training total size : "+str(df[df['currentTraining']==1]['currentTraining'].count()))
    return df


#build image dataset according to data
def buildImageset(df):
    start=time.time()
    sample_size = df['Image Index'].count()
    print("buildImageset sample_size : "+str(sample_size))
    Y = np.ndarray((sample_size,NUMBER_OF_DESEASES), dtype=np.float32)
    X = np.ndarray((sample_size, img_width, img_height, 1), dtype=np.float32)
    pat_list = ['Cardiomegaly','Emphysema','Effusion','Hernia','Nodule','Pneumothorax','Atelectasis','Pleural_Thickening','Mass','Edema','Consolidation','Infiltration','Fibrosis','Pneumonia']

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
        Y[index] = row[pat_list]

    end=time.time()
    print("loop build images set duration :" +str(end-start)+"sec")
    print("X : "+str(X.shape))
    print("Y : "+str(Y.shape))
    return X, Y

def history_plot(history):
    plt.figure(1)

    plt.subplot(211)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.savefig("current.png")
    #plt.close(fig)

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
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 120:
        lr *= 1e-3
    elif epoch > 80:
        lr *= 1e-2
    elif epoch > 60:
        lr *= 1e-1
    print("Learning rate: "+ str(lr))
    return lr


################################################
############### MAIN ###########################
################################################

#Load model
model = loadModel()
# SGD > RMSprop > Adam
#sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
opt = Adam(lr=lr_schedule(0))
#optimizer = RMSprop(lr=lr_schedule(0), decay=1e-6)
model.compile(loss='binary_crossentropy',
#model.compile(loss=weighted_binary_crossentropy,
              optimizer=opt,
              metrics=['accuracy'])

#create or load dataset
data = createDataSet()

#load and build training set
data = loadTrainingDataset(data)
dataTrain = data[data['currentTraining']==1].reset_index()
X_train, Y_train = buildImageset(dataTrain)

#Add callback to monitor model quality
filepath=MODEL_NAME+"-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)
callbacks_list = [checkpoint,lr_scheduler,lr_reducer]
#callbacks_list = [checkpoint]

#train model
start=time.time()
history=model.fit(X_train, Y_train, batch_size=batch_size, epochs=epoch, shuffle=True, callbacks=callbacks_list, validation_split=0.05, verbose=1)
#history=model.fit(X_train, Y_train, batch_size=batch_size, epochs=epoch, shuffle=True, validation_split=0.1, verbose=1)
end=time.time()
print("fit duration :" +str(end-start)+"sec")

#save model
model.save(MODEL_NAME)


#set currentTraining to O and Trained to 1
data.loc[data['currentTraining']==1,['trained']]=1
print("Already trained images : "+str(data[data['trained']==1]['Image Index'].count()))
#save current step to training file 
data.to_csv(SAVED_FILE,index=False)

#print(history.history)
history_plot(history)



