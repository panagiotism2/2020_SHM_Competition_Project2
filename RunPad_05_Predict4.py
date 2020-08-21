# CLEAR ALL
from IPython import get_ipython
get_ipython().magic('reset -sf')
  
# Import libraries
from utils2 import get_predictions
import scipy.io as sio
import pandas as pd
import numpy as np
import os

def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles


# In[]: Define the number of channels for each 
n_channels=38

# In[] Get List of all data names in the folder RawData_MAT
dirName  = "C:/Users/martakip/Documents/Panagiotis/Projects/2020_SHM_competition/Python/RawData_MAT" 
listOfFiles = getListOfFiles(dirName)
n_data=len(listOfFiles)
listOfFilenames=[]
n_data=len(listOfFiles)

# Prepare row names for dataframe
for i in range(n_data):
    temp=listOfFiles[i].split("\\")[1]
    temp=temp.split(".")[0]
    listOfFilenames.append(temp)
    
# Preallocate dataframes
ch_vector=np.arange(1,n_channels+1)
df_predictions=pd.DataFrame(np.zeros(shape=(n_data,n_channels)), columns=ch_vector,index=listOfFilenames) 
df_probabilities=pd.DataFrame(np.zeros(shape=(n_data,n_channels)), columns=ch_vector,index=listOfFilenames) 
# In[]: Get Predictions and store them in dataframes. A dataframe with the corresponing probabilities of the prediction is also assembled.
for i in range(n_data):
    #file_path="C:/Users/martakip/Desktop/SHAP_test/data_in/MAT/2012-01-01 18-VIB.mat"
    file_path=listOfFiles[i]
    # Convert mat to numpy data
    mat_contents = sio.loadmat(file_path)
    data_raw=mat_contents['data'] # Should be an 1D or 2D array with structure: samples x channels
    
    
    
    # Get Predictions
    class_prediction,pred_probabilty=get_predictions(data_raw)
    
    df_predictions.iloc[i,:]=class_prediction
    df_probabilities.iloc[i,:]=pred_probabilty
    

print(df_predictions)

