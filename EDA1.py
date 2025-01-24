#!/usr/bin/env python
# coding: utf-8

# In[3]:


#load the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


data = pd.read_csv("data_clean.csv")
print(data)


# In[7]:


data.info()


# In[8]:


# DataFrame attributes
print(type(data))
print(data.shape)
print(data.size)


# In[10]:


# Drop duplicate column( Temp c)and Unnamed column
#data1 = data.drop(['Unnamed: 0',"Temp C"], axis =1, inplace = True)

data1 = data.drop(['Unnamed: 0',"Temp C"], axis =1)
data1


# In[12]:


# Convert the month column data type to float data type

data1['Month']=pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[14]:


# Print all duplicated rows
data1[data1.duplicated(keep = False)]


# In[15]:


## Checking for duplicated rows in the table
# print only the duplicated row(one) only
data1[data1.duplicated()]


# In[16]:


# Drop duplicated rows
data1.drop_duplicates(keep='first', inplace = True)
data1


# In[ ]:




