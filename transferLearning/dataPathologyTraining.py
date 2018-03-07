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
from keras.layers import Conv2D, MaxPooling2D, Input, AveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import SGD,Adagrad,Adadelta,RMSprop,Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from keras.regularizers import l2
from keras import backend as K

DIRECT_ORIGINALS='../../images/'
DIRECT_AUGMENTED='../../imagesAugmented/'
MAIN_FILE='Data_Entry_2017.csv'
AUGMENTED_FILE='Data_augmented.csv'
MODEL_NAME="myModel.h5"
DATASET_SIZE=5000


batch_size=8
epoch=100
NUMBER_OF_DESEASES=1
img_width, img_height = 224 , 224
input_shape = (img_width, img_height,1)
#cropping dimension
crop_x,crop_y,crop_w,crop_h=(112,112,800,800)

# Subtracting pixel mean improves accuracy
subtract_pixel_mean = True


np.random.seed(1234)


# Model parameter
# ----------------------------------------------------------------------------
#           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
#           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
# ----------------------------------------------------------------------------
# ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
# ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
# ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
# ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
# ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
# ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
# ---------------------------------------------------------------------------
n = 3

# Model version
# Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
version = 2

if version == 1:
    depth = n * 6 + 2
elif version == 2:
    depth = n * 9 + 2

# Model name, depth and version
model_type = 'ResNet%dv%d' % (depth, version)



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

#Create datasets
def loadTrainingDataset(df):
    #build dataset to train
    df['currentTraining']=0

    total_size = df[(df[PATHOLOGY_NAME]==1)&(df['test']==0)&(data['trained']==0)][PATHOLOGY_NAME].count()
    total_size = min(DATASET_SIZE,total_size)
    df.loc[df[(df[PATHOLOGY_NAME]==1)&(df['test']==0)&(data['trained']==0)].sample(total_size).index,['currentTraining']]=1
    print(PATHOLOGY_NAME+" train size : "+str(total_size))

    #add no pathology
    df.loc[df[(df[PATHOLOGY_NAME]==0)&(df['test']==0)&(data['trained']==0)].sample(total_size).index,['currentTraining']]=1

    return df

#build image dataset according to data
def buildImageset(df):
    start=time.time()
    sample_size = df['Image Index'].count()
    print("buildImageset : "+PATHOLOGY_NAME+" sample_size : "+str(sample_size))
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
        #X[index] = (img - img.min())/(img.max() - img.min())
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
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            activation-bn-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, num_classes=10):
    """ResNet Version 1 Model builder [a]
    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_v2(input_shape, depth, num_classes=10):
    """ResNet Version 2 Model builder [b]
    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


if version == 2:
    model = resnet_v2(input_shape=input_shape, depth=depth, num_classes=2)
else:
    model = resnet_v1(input_shape=input_shape, depth=depth, num_classes=2)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])
model.summary()
print(model_type)


#load and build training set
data = createDataSet()
data = loadTrainingDataset(data)
dataTrain = data[data['currentTraining']==1].reset_index()
X_train, Y_train = buildImageset(dataTrain)

# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(X_train, axis=0)
    X_train -= x_train_mean
    
filepath=MODEL_NAME+"-{val_acc:.2f}.hdf5"

#Add callback to monitor model quality
# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks_list = [checkpoint, lr_reducer, lr_scheduler]


#train model
Y_train = to_categorical(Y_train, num_classes=2)
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
