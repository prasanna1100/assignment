#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
df = pd.read_csv("universities.csv")
df


# In[3]:


np.mean(df["SAT"])


# In[8]:


np.median(df["SAT"])


# In[9]:


# Visualize the GradeRate using histogram
import matplotlib.pyplot as plt
import seaborn as sns


# In[11]:


plt.figure(figsize=(6,3))
plt.title("Acceptance Ratio")
plt.hist(df["Accept"])


# In[12]:


sns.histplot(df["Accept"], kde =True)


# Observations
#  -> In Acceptance ratio the data distribution is non-symmetrical and right skewed

# In[ ]:




