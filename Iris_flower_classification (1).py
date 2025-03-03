#!/usr/bin/env python
# coding: utf-8

# In[52]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[53]:


iris = pd.read_csv("iris.csv")
iris


# In[54]:


iris.info()


# In[55]:


iris["variety"].value_counts()


# In[56]:


# Print allduplicated rows
iris[iris.duplicated(keep= False)]


# #### Observations
# - There are 150 rows and 5 columns
# - There are no null values 
# - There are no duplicated values 
# - The x-columns are sepal.length, sepal.width, petal.length, petal.width
# - All the x-columns are continuous
# - The y-column is "variety" which is categorical
# - There are three flower categories (classes)

# In[57]:


iris = iris.drop_duplicates(keep='first')


# In[58]:


iris[iris.duplicated]


# In[59]:


#reset the indexx
iris=iris.reset_index(drop=True)
iris


# In[60]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
iris.iloc[:,-1]=labelencoder.fit_transform(iris.iloc[:,-1])
iris.head()


# In[61]:


# Check the data types after label encoding
iris.info()


# #### Observation
# - The target column(variety) is still object type.It needs to to be converted to numeric(int)

# In[62]:


# Convert the target column data type to int
iris['variety'] = pd.to_numeric(labelencoder.fit_transform(iris['variety']))
print(iris.info())


# In[63]:


# Divide the dataset into x-columns and y- columns
X=iris.iloc[:,0:4]
Y=iris['variety']


# In[64]:


Y


# In[65]:


X


# In[66]:


# Further splitting of data into training and testing data sets
x_train, x_test,y_train,y_test = train_test_split(X,Y, test_size=0.3,random_state = 1)
x_train.head(20)

