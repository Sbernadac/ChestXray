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
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import SGD,Adagrad,Adadelta,RMSprop,Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from keras.applications.vgg19 import VGG19


DIRECT_ORIGINALS='../../images/'
DIRECT_AUGMENTED='../../imagesAugmented/'
MAIN_FILE='Data_Entry_2017.csv'
AUGMENTED_FILE='Data_augmented.csv'
MODEL_NAME="myModel.h5"
DATASET_SIZE=5000
FIT_AGAIN=True

batch_size=16
epoch=100
NUMBER_OF_DESEASES=1
img_width, img_height = 224 , 224
input_shape = (img_width, img_height,3)
#cropping dimension
crop_x,crop_y,crop_w,crop_h=(112,112,800,800)

np.random.seed(1234)

#get input data
PATHOLOGY_NAME = raw_input("Please enter pathology name : ")
if len(PATHOLOGY_NAME)==0:
    PATHOLOGY_NAME='Cardiomegaly'

MODEL_NAME="my"+PATHOLOGY_NAME+".h5"
FILE_NAME='Data_'+PATHOLOGY_NAME+'.csv'
result = [PATHOLOGY_NAME]

def createDataSet():
    #check if training DataFrame already exists
    if os.path.exists(FILE_NAME):
        data = pd.read_csv(FILE_NAME)
    #create new training DataFrame
    else:
        #import original image data set description
        dfo = pd.read_csv(MAIN_FILE)
        dfo = dfo[['Image Index','Finding Labels']]
        dfo[PATHOLOGY_NAME] = dfo['Finding Labels'].apply(lambda x: 1 if PATHOLOGY_NAME in x else 0)
        dfo = dfo.drop(['Finding Labels'],axis=1)
        #add columns for augmented type
        dfo['Augmented'] = 0

        #import augmented image data set description
        dfa = pd.read_csv(AUGMENTED_FILE)
        dfa = dfa[['Image Index',PATHOLOGY_NAME]]
        dfa['Augmented'] = 1


        #concatenate dataframes
        data = pd.concat([dfo,dfa], ignore_index=True)

        #Add columns for train/validation, tests and already trained:
        data['test']=0
        data['trained']=0

        #we have 112120 Xrays, and we want 95% for train (90% for training and 5% for validation) and 5% for test
        #we also need to have same deseases distribution for all of them
	data.loc[data[(data[PATHOLOGY_NAME]==1)&(data['Augmented']==0)].sample(frac=0.2).index,['test']]=1
        count = data[data['test']==1]['Image Index'].count()
        data.loc[data[(data[PATHOLOGY_NAME]==0)&(data['Augmented']==0)].sample(count).index,['test']]=1
        
    print("Training and validation dataset size : "+str(data[data['test']==0]['Image Index'].count()))
    print("Test dataset size : "+str(data[data['test']==1]['Image Index'].count()))
    return data

#loadModel : create or load already trained model
def loadModel():
    if os.path.exists(MODEL_NAME):
        model = load_model(MODEL_NAME)
    else:
        base_model = VGG19(include_top=False, weights='imagenet', pooling='avg')

        # add a global spatial average pooling layer
        x = base_model.output

        #x = GlobalAveragePooling2D()(x)
        x = Dense(2048, activation='relu')(x)
        x = Dropout(0.4)(x)
        # let's add a fully-connected layer
        x = Dense(2048, activation='relu')(x)
        x = Dropout(0.3)(x)
        # and a logistic layer -- 1 classes
        predictions = Dense(1, activation='sigmoid')(x)

        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)

        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutionals layers
        for layer in base_model.layers:
            layer.trainable = False

        # compile the model (should be done *after* setting layers to non-trainable)
        #model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

        #optimizer = RMSprop(lr=0.001, decay=1e-6)
        #opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.0)
        #sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
        #model.compile(loss='binary_crossentropy',
              #optimizer=sgd,
              #metrics=['accuracy'])

    print(model.summary())
    return model

#Create datasets
def loadTrainingDataset(df):
    #build dataset to train
    df['currentTraining']=0
    total_size = df[(df[PATHOLOGY_NAME]==1)&(df['test']==0)&(df['trained']==0)][PATHOLOGY_NAME].count()
    total_size = min(DATASET_SIZE,total_size)
    df.loc[df[(df[PATHOLOGY_NAME]==1)&(df['test']==0)&(df['trained']==0)].sample(total_size).index,['currentTraining']]=1
    print(PATHOLOGY_NAME+" train size : "+str(total_size))

    #add no pathology
    df.loc[df[(df[PATHOLOGY_NAME]==0)&(df['test']==0)&(df['trained']==0)].sample(total_size).index,['currentTraining']]=1

    return df

#build image dataset according to data
def buildImageset(df):
    start=time.time()
    sample_size = df['Image Index'].count()
    print("buildImageset : "+PATHOLOGY_NAME+" sample_size : "+str(sample_size))
    Y = np.ndarray((sample_size,1), dtype=np.float32)
    X = np.ndarray((sample_size, img_width, img_height, 3), dtype=np.float32)

    #import images as array
    for index, row in df.iterrows():
        if row['Augmented']==1:
            d = DIRECT_AUGMENTED
        else:
            d = DIRECT_ORIGINALS
        # Load image in grayscale
        img = cv2.imread(d+row['Image Index'],1)
        #crop image
        img = img[crop_x:crop_x+crop_w,crop_y:crop_y+crop_h]
        img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_AREA)
        #img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = img_to_array(img)
        X[index] = img.astype('float32') / 255
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
    lr = 1e-3
    if epoch > 100:
        lr *= 0.5e-3
    elif epoch > 80:
        lr *= 1e-3
    elif epoch > 60:
        lr *= 1e-2
    elif epoch > 40:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


#Load model
model = loadModel()
model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])

#load and build training set
data = createDataSet()
data = loadTrainingDataset(data)

dataTrain = data[data['currentTraining']==1].reset_index()
X_train, Y_train = buildImageset(dataTrain)
x_train_mean = np.mean(X_train, axis=0)
X_train -= x_train_mean

#Add callback to monitor model quality
filepath=MODEL_NAME+"-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks_list = [checkpoint,lr_reducer,lr_scheduler]

#train model
#Y_train = to_categorical(Y_train, num_classes=2)
start=time.time()
history=model.fit(X_train, Y_train, batch_size=batch_size, epochs=epoch, shuffle=True, callbacks=callbacks_list, validation_split=0.10, verbose=1)
end=time.time()
print("fit duration :" +str(end-start)+"sec")

#save model
model.save(MODEL_NAME)

#set currentTraining to O and Trained to 1
data.loc[data['currentTraining']==1,['trained']]=1
print("Already trained images : "+str(data[data['trained']==1]['Image Index'].count()))
#save current step to training file 
data.to_csv(FILE_NAME,index=False)

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

history_plot(history)
