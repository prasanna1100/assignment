#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans


# ##### Clustering-Divide 

# In[10]:


Univ = pd.read_csv("Universities.csv")
Univ


# In[11]:


Univ.info()


# In[13]:


Univ.describe()


# In[14]:


Univ.isna().sum()


# #### Standardization of the Data

# In[26]:


# Read all numeric columns in to Univ1
Univ1 = Univ.iloc[:,1:]


# In[27]:


Univ1


# In[30]:


cols = Univ1.columns


# In[31]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_Univ_df = pd.DataFrame(scaler.fit_transform(Univ1),columns = cols)
scaled_Univ_df


# In[32]:


# Build Cluster algorithm
from sklearn.cluster import KMeans
clusters_new = KMeans(3, random_state=0)
clusters_new.fit(scaled_Univ_df)


# In[33]:


# Print the cluster labels
clusters_new.labels_


# In[34]:


set(clusters_new.labels_)


# In[35]:


# Assign clusters to the Univ data set
Univ['clusterid_new'] = clusters_new.labels_


# In[36]:


Univ


# In[37]:


Univ[Univ['clusterid_new']==1]


# In[39]:


# USe gropuby to find aggregated (mean) values in each cluster
Univ.iloc[:,1:].groupby("clusterid_new").mean()


# #### Observations 
# - Cluster 2 appears to be the top rated universities cluster a sthe cut off Score, Top10, SFRatio parameter mean values are higher
# - Cluster 1 appears to occupy the middle level rated universities
# - Cluster 0 comes as the lower level rated universities

# In[40]:


Univ[Univ['clusterid_new']==0]


# #### Finding optimal k value using elbow plot

# In[42]:


wcss = []
for i in range(1, 20):
    
    kmeans = KMeans(n_clusters=i,random_state=0 )
    kmeans.fit(scaled_Univ_df)
    #Kmeans.fit(Univ1)
    wcss.append(kmeans.inertia_)
print(wcss)
plt.plot(range(1, 20), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[ ]:




