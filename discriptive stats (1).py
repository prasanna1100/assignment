#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
df = pd.read_csv("universities.csv")
df


# In[2]:


np.mean(df["SAT"])


# In[3]:


np.median(df["SAT"])


# In[4]:


# Visualize the GradeRate using histogram
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


plt.figure(figsize=(6,3))
plt.title("Acceptance Ratio")
plt.hist(df["Accept"])


# In[6]:


sns.histplot(df["Accept"], kde =True)


# Observations
#  -> In Acceptance ratio the data distribution is non-symmetrical and right skewed

# In[7]:


# Visualization using boxplot
#Create a pandas series of batsman1 scores
s1 = [20,15,10,25,30,35,28,40,45,60]
scores1 = pd.Series(s1)
scores1


# In[8]:


plt.boxplot(scores1, vert=False)


# In[14]:


plt.figure(figsize=(6,2))
plt.title("Boxplot for batsman scores")
plt.xlabel("Scores")
plt.boxplot(scores1, vert=False)


# In[19]:


# Add extreme values to scores and plot the boxplot
s2 = [20,15,10,25,30,35,28,40,45,60,120,150]
scores2 = pd.Series(s2)
print(scores2)

plt.figure(figsize=(6,2))
plt.title("Boxplot for batsman scores")
plt.xlabel("Scores")
plt.boxplot(scores2, vert=False)


# In[ ]:





# In[ ]:




