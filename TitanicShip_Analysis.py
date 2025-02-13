#!/usr/bin/env python
# coding: utf-8

# In[9]:


get_ipython().system('pip install mlxtend')


# In[13]:


import pandas as pd
import mlxtend
from mlxtend.frequent_patterns import apriori,association_rules
import matplotlib.pyplot as plt


# In[14]:


titanic = pd.read_csv("Titanic.csv")
titanic


# In[15]:


titanic.info()


# In[ ]:


#### Observations
- There are no null values
- All columns are


# In[16]:


counts = titanic['Class'].value_counts()
plt.bar(counts.index, counts.values)


# In[17]:


df = pd.get_dummies(titanic, dtype=int)
df.head()


# In[18]:


df.info()


# In[19]:


frequent_itemsets=apriori(df,min_support=0.05,use_colnames=True,max_len=None)
frequent_itemsets


# In[21]:


# Generate association rules with metrics
rules = association_rules(frequent_itemsets, metric="lift",min_threshold=1.0)
rules


# In[22]:


rules.sort_values(by='lift', ascending = False)


# In[ ]:


#### Conclusion
- Adult Females travelling in 1st class were among the most survived


# In[25]:


import matplotlib.pyplot as plt
rules[['support','confidence','lift']].hist(figsize=(15,7))
plt.show()


# In[26]:


import matplotlib.pyplot as plt

plt.scatter(rules['support'], rules['confidence'])
plt.show()


# #### The confidence value is increasing with increase in support for most 

# In[28]:


plt.scatter(rules['confidence'], rules['lift'])
plt.show()


# In[29]:


rules[rules["consequents"]== ({"Survived_Yes"})]


# In[ ]:




