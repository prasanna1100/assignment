#!/usr/bin/env python
# coding: utf-8

# In[1]:


#load the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv("data_clean.csv")
print(data)


# In[3]:


data.info()


# In[4]:


# DataFrame attributes
print(type(data))
print(data.shape)
print(data.size)


# In[5]:


# Drop duplicate column( Temp c)and Unnamed column
#data1 = data.drop(['Unnamed: 0',"Temp C"], axis =1, inplace = True)

data1 = data.drop(['Unnamed: 0',"Temp C"], axis =1)
data1


# In[6]:


# Convert the month column data type to float data type

data1['Month']=pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[7]:


# Print all duplicated rows
data1[data1.duplicated(keep = False)]


# In[8]:


## Checking for duplicated rows in the table
# print only the duplicated row(one) only
data1[data1.duplicated()]


# In[9]:


# Drop duplicated rows
data1.drop_duplicates(keep='first', inplace = True)
data1


# Rename the columns

# In[10]:


# Change column names(Rename the columns)
data1.rename({'Solar.R': 'Solar'}, axis=1, inplace = True)
data1


# Impute the missing values

# In[11]:


data.info()


# In[12]:


# Display data1 missing values count in each column using isnull().sum()
data1.isnull().sum()


# In[13]:


# visualize data1 missing values using graph

cols = data1.columns
colors = ['black', 'white']
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colors),cbar = True)


# In[14]:


# Find the mean and median values of each numeric column
# Imputation of missing value with median
median_ozone = data1["Ozone"].median()
mean_ozone = data1["Ozone"].mean()
print("Meadian of Ozone: ", median_ozone)
print("Mean of Ozone: ", mean_ozone)


# In[15]:


# Replace the Ozone missing values with median value
data['Ozone'] = data1['Ozone'].fillna(median_ozone)
data1.isnull().sum()


# In[16]:


# print the data1 5 rows
data1.head()


# In[18]:


# Find the mode values of categorical column (weather)

print(data1["Weather"].value_counts())
mode_weather = data1["Weather"].mode()[0]
print(mode_weather)


# In[19]:


# Impute missing values (Replace NaN with mode etc.) using filna()
data1["Weather"] = data1["Weather"].fillna(mode_weather)
data1.isnull().sum()


# In[ ]:




