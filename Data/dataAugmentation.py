# coding: utf-8
import numpy as np
import pandas as pd
import os
import sys
import cv2
import time
import keras
import commands
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
 
#DEFAULT SIZE:
#Hernia size=227
#Pneumonia size=1353
#Fibrosis size=1686
#Edema size=2303
#Emphysema size=2516
#Cardiomegaly size=2772
#Pleural_Thickening size=3385
#Consolidation size=4667
#Pneumothorax size=5298
#Mass size=5746
#Nodule size=6323
#Atelectasis size=11535
#Effusion size=13307
#Infiltration size=19870

#MAIN DIRECTORY
MAIN_DIRECTORY='./train/'
IMAGES_DIRECTORY='../../../../images/'
IMAGES_AUGMENTED_DIRECTORY='../imagesAugmented/'
DATASET_FILE='Data_Entry_Training.csv'

DESEASE_SIZE=6000

#create directories for augmented images files
def createDirectories():

    start=time.time()

    #import Test Dataset to avoid augmentation for those files
    df = pd.read_csv(DATASET_FILE)

    groupList = df['Finding Labels'].unique()
    #remove 'No Finding':
    groupList = [ x for x in groupList if x != 'No Finding' ]
    #As we have large number of Infiltration, Effusion and Atelectasis we dont keep them
    cleaned_list = [ x for x in groupList if 'Infiltration' not in x and 'Effusion' not in x and 'Atelectasis' not in x ]

    #create directory for each group
    for group in cleaned_list:
        group=group.replace('|','-')
        os.system('mkdir -p '+MAIN_DIRECTORY+group+'/originals')
        os.system('mkdir -p '+MAIN_DIRECTORY+group+'/augmented')

    end=time.time()
    print("createDirectories duration :" +str(end-start)+" sec")

#create original image links, except for test set
def createImageLinks():
    
    start=time.time()

    #import Test Dataset to avoid augmentation for those files
    df = pd.read_csv(DATASET_FILE)

    desease_list = ['Hernia','Pneumonia','Fibrosis','Edema','Emphysema','Cardiomegaly','Pleural_Thickening','Consolidation','Pneumothorax','Mass','Nodule']

    #Add linked column
    df['linked']=0

    #dont keep test image 
    df=df[df['test']==0].reset_index()

    for desease in desease_list :
        #just keep one desease
        print("createImageLinks number of data : " +str(df[(df[desease]==1)&(df['linked']==0)][desease].count()))
        for index, row in df[(df[desease]==1)&(df['linked']==0)].iterrows():
            direct = row['Finding Labels'].replace('|','-')
            if ('Effusion' not in direct) and ('Infiltration' not in direct) and ('Atelectasis' not in direct):
                files = row['Image Index']
                os.system('ln -s '+IMAGES_DIRECTORY+files+' '+MAIN_DIRECTORY+direct+'/originals')
        df.loc[df[desease]==1,['linked']]=1

    end=time.time()
    print("createImageLinks duration :" +str(end-start)+" sec")


#create augmented images
def createAugmented(pat,fracW=0.2, fracH=0.2, rot=20, z=0.2):

    #import Test Dataset to avoid augmentation for those files
    df = pd.read_csv(DATASET_FILE)

    start=time.time()
    #create image generator
    datagen = ImageDataGenerator(horizontal_flip=False, vertical_flip=False,
          data_format='channels_last',width_shift_range=fracW,height_shift_range=fracH,
          rotation_range=rot,zoom_range=z)


    #just keep PATHOLOGY_NAME
    data=df[df[pat]==1]

    groupList = data['Finding Labels'].unique()
    #As we have large number of Infiltration, Effusion and Atelectasis
    #We will just augment pathology in other cases:
    groupList = [ x for x in groupList if 'Infiltration' not in x and 'Effusion' not in x and 'Atelectasis' not in x ]

    #check for number of images only belonging to current pathology
    status, output = commands.getstatusoutput('ls -alrt train/*'+pat+'*/originals/* | wc -l')
    batch_size=int(output)
    print("batch_size : "+str(batch_size))
    RATIO = min(20,(DESEASE_SIZE-batch_size)/batch_size)
    print("ratio : "+str(RATIO))

    for group in groupList:
        i = 1
        group=group.replace('|','-')
        #check if this group has already been augmented:
        status,output = commands.getstatusoutput('ls -alrt '+MAIN_DIRECTORY+group+'/augmented/* | wc -l')
        if len(output)>7:
            #check if this group contains images:
            status,output = commands.getstatusoutput('ls -alrt '+MAIN_DIRECTORY+group+'/originals/* | wc -l')
            if len(output)<7:
                print("Current group :"+group)
                for batch in datagen.flow_from_directory(MAIN_DIRECTORY+group, target_size=(1024,1024), batch_size=batch_size,
                    color_mode='grayscale', save_to_dir=MAIN_DIRECTORY+group+'/augmented', save_prefix=group, seed=1, follow_links=True):
                    i += 1
                    if i > int(RATIO):
                        break  # otherwise the generator would loop indefinitely
        else:
            print("Group "+group+" already contains augmented images: "+ output)

    end=time.time()
    print("createAugmented duration :" +str(end-start)+" sec")

#create DataFrame with all augmented images
def createAugmentedDataFrame():
    start=time.time()
    #import images as array to create huge Dataframe
    data = pd.DataFrame(columns=['Image Index','Finding Labels','No Finding','Cardiomegaly','Emphysema','Effusion','Hernia','Nodule','Pneumothorax','Atelectasis','Pleural_Thickening','Mass','Edema','Consolidation','Infiltration','Fibrosis','Pneumonia','test','Augmented'])
    desease_list=['Cardiomegaly','Emphysema','Effusion','Hernia','Nodule','Pneumothorax','Atelectasis','Pleural_Thickening','Mass','Edema','Consolidation','Infiltration','Fibrosis','Pneumonia']
    
    #import Test Dataset
    df = pd.read_csv(DATASET_FILE)
    #Add augmented column
    df['Augmented'] = 0

    i=0
    for dire in os.listdir(MAIN_DIRECTORY):
	row=['','']+[0]*17
	j=3
	for desease in desease_list:
		if desease in dire:
			row[j]=1
		j+=1
	for files in os.listdir(MAIN_DIRECTORY+dire+'/augmented'):
		row[0]=files
		data.loc[i]=row	
		i+=1
                
    #Add augmented column
    data['Augmented'] = 1

    df_total = pd.concat([df,data], ignore_index=True)
    df_total = df_total.drop(['Finding Labels'],axis=1)

    df_total.to_csv('Data_augmented.csv',index=False)
    end=time.time()
    print("createAugmentedDataFrame duration :" +str(end-start)+" sec")

def createAugmentedLinks():
    start=time.time()
    for dire in os.listdir(MAIN_DIRECTORY):
        for files in os.listdir(MAIN_DIRECTORY+dire+'/augmented'):
            #print('ln -s ../imagesDirectory/'+MAIN_DIRECTORY+dire+'/augmented/'+files+' '+IMAGES_AUGMENTED_DIRECTORY)
            os.system('ln -s ../imagesDirectory/'+MAIN_DIRECTORY+dire+'/augmented/'+files+' '+IMAGES_AUGMENTED_DIRECTORY)
    end=time.time()
    print("createAugmentedLinks duration :" +str(end-start)+" sec")


######
#MAIN#
######



#get user main action:
ACTION = raw_input("Do you want to build directories for augmented images [y/N] : ")
if (ACTION=='y'):
        createDirectories()
        sys.exit()
ACTION = raw_input("Do you want to link originals images [y/N] : ")
if (ACTION=='y'):
        createImageLinks()
        sys.exit()
ACTION = raw_input("Do you want to create augmented dataframe [y/N] : ")
if (ACTION=='y'):
        createAugmentedDataFrame()
        sys.exit()
ACTION = raw_input("Do you want to create augmented image links to main image directory [y/N] : ")
if (ACTION=='y'):
        createAugmentedLinks()
        sys.exit()
ACTION = raw_input("Do you want to create augmented images [Y/n] : ")
if (ACTION=='n'):
        sys.exit()

        
#get pathology name from user
#PATHOLOGY_NAME = raw_input("Please enter pathology name : ")
#import Test Dataset to avoid augmentation for those files
df = pd.read_csv(DATASET_FILE)

desease_list = ['Hernia','Pneumonia','Fibrosis','Edema','Emphysema','Cardiomegaly','Pleural_Thickening','Consolidation','Pneumothorax','Mass','Nodule']

for desease in desease_list :
    count = df[(df[desease]==1)&(df['test']==0)][desease].count()
    print("Number of original images : "+str(count))

    #compute current number of augmented images
    status, augmented = commands.getstatusoutput('ls -alrt '+MAIN_DIRECTORY+'*'+desease+'*/augmented/* | wc -l')
    if len(augmented)>7:
        augmented='0'
    print("Number of augmented images currently added : "+augmented)

    #compute number of originals images that may be augmented
    status, originals = commands.getstatusoutput('ls -alrt '+MAIN_DIRECTORY+'*'+desease+'*/originals/* | wc -l')
    if augmented=='0':
        print("Number of original images to augment: "+originals)
    else:
        #we have to compute how many original images are not already been augmented
        groupList = df['Finding Labels'].unique()
        #remove 'No Finding':
        groupList = [ x for x in groupList if x != 'No Finding' ]
        #As we have large number of Infiltration, Effusion and Atelectasis we dont keep them
        groupList = [ x for x in groupList if 'Infiltration' not in x and 'Effusion' not in x and 'Atelectasis' not in x ]
        #just keep groups that contain current pathology
        cleaned_list = [ x for x in groupList if desease in x ]
        count=0
        for group in cleaned_list:
            group=group.replace('|','-')
            status, augmented = commands.getstatusoutput('ls -alrt '+MAIN_DIRECTORY+group+'/augmented/* | wc -l')
        #print("Current group : "+group+" nb aumented : "+augmented)
            if len(augmented)<7:
                status, output = commands.getstatusoutput('ls -alrt '+MAIN_DIRECTORY+group+'/originals/* | wc -l')
            #print("     nb originals : \n"+output)
                count+=int(output)
        result=int(originals)-count
        print("Number of remaining original images to augment: "+str(result))

#get wanted ratio from user
#RATIO = raw_input("Please enter wanted ratio of augmented images (0 to stop): ")
#if int(RATIO)==0:
#    sys.exit()

#get fraction for width_shift_range and width_shift_range:
#frac_width = raw_input("Please enter value for width shift range (default 0.1): ")
#if len(frac_width) > 0:
#    fracW = float(frac_width)
#frac_height = raw_input("Please enter value for height shift range (default 0.1): ")
#if len(frac_height) > 0:
#    fracH = float(frac_height)
#rotation = raw_input("Please enter value rotation (default 10): ")
#if len(rotation) > 0:
#    rot = int(rotation)
#zoom = raw_input("Please enter value zoom range (default 0.1): ")
#if len(zoom) > 0:
#    z = float(zoom)

    createAugmented(desease)

