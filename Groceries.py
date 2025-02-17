#!/usr/bin/env python
# coding: utf-8

# In[9]:


get_ipython().system('pip install mlxtend')


# In[18]:


import pandas as pd
import mlxtend
from mlxtend.frequent_patterns import apriori,association_rules
import matplotlib.pyplot as plt


# In[20]:


import pandas as pd
import numpy as np
df = pd.read_csv("Groceries_dataset.csv")
df


# In[27]:


import pandas as pd

df = pd.read_csv("groceries_dataset.csv")
df.info()


# In[28]:


print(df.head())


# In[29]:


print("Columns:", df.columns.tolist())


# In[32]:


counts = df['itemDescription'].value_counts()
plt.bar(counts.index, counts.values)


# In[34]:


df = pd.get_dummies(df, dtype=int)
df.head()


# In[ ]:




