# coding: utf-8
import numpy as np
import pandas as pd
import os
import time


MAIN_FILE='Data_Entry_2017.csv'
SAVED_FILE='Data_Entry_Training.csv'


np.random.seed(1234)


#columns for each desease
pathology_list = ['Cardiomegaly','Emphysema','Effusion','Hernia','Nodule','Pneumothorax','Atelectasis','Pleural_Thickening','Mass','Edema','Consolidation','Infiltration','Fibrosis','Pneumonia','No Finding']

def createTestDataSet():
    #create new training DataFrame
    #import original image data set description
    data = pd.read_csv(MAIN_FILE)
    data = data[['Image Index','Finding Labels']]
    for pathology in pathology_list :
        data[pathology] = data['Finding Labels'].apply(lambda x: 1 if pathology in x else 0)

    #Add columns for train/validation, tests and already trained:
    data['test']=0

    #we have 112120 Xrays, and we want 95% for train (90% for training and 5% for validation) and 5% for test
    #we also need to have same deseases distribution for all of them
    deseases={}
    for pathology in pathology_list :
	deseases[pathology]=data[data[pathology]==1][pathology].count()
    #sort deseases
    sort_deseases=sorted(deseases.items(), key=lambda x: x[1])

    #select test set
    for pat,num in sort_deseases:
        #select desease sample, need to take care of multiple deseases
        count = num/10  #take 10%
        count = count - data[(data['test']==1)&(data[pat]==1)][pat].count() #remove images already selected for test
        if count > 0:
            print("take "+str(count)+" for "+pat)
	    data.loc[data[(data[pat]==1)&(data['test']==0)].sample(count).index,['test']]=1

    print("Training and validation dataset size : "+str(data[data['test']==0]['Image Index'].count()))
    print("Test dataset size : "+str(data[data['test']==1]['Image Index'].count()))
    return data



#create test dataset
data = createTestDataSet()

#save current dataset 
data.to_csv(SAVED_FILE,index=False)




