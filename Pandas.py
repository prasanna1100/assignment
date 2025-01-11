#!/usr/bin/env python
# coding: utf-8

# Create pandas series

# In[2]:


# Import pandas
import pandas as pd


# In[3]:


# Create pandas series using list 

data = [10,20,30,40]
series = pd.Series(data)
print(series)


# In[4]:


# Create series using a custom index
data = [1, 2, 3, 4]
i = ['A', 'B', 'C', 'D']
series = pd.Series(data, index=i)
print(series)


# In[ ]:


Pandas series objects are size - immutable
But it allows to modify element value


# In[5]:


# Create pandas series using dictionary
data = {'a': 10, 'b': 20, 'c': 30}
series = pd.Series(data)
print(series)


# In[6]:


series.replace(20,40)


# In[8]:


# create series using numpy array
import numpy as np

data = np.array([100, 200, 300])
series = pd.Series(data, index=['a','b','c',])
print(series)


# In[ ]:


Pandas dataframe


# In[13]:


# Create pandas dataframe from dictionary of lists
import pandas as pd

data = {'Name': ['Alice', 'Bob','Mary'], 'Age':[25,30,68], 'Country':["USA","UK","AUS"]}
df = pd.DataFrame(data)
print(df)


# In[14]:


# Create pandas dataframe from numpy array
import numpy as np

array = np.array([[1, 2,3], [4, 5, 6],[7, 8, 9]])
print(array)
df = pd.DataFrame(array, columns=['A', 'B', 'C'])
print(df)


# In[ ]:




