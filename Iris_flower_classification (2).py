#!/usr/bin/env python
# coding: utf-8

# In[55]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


# In[56]:


iris = pd.read_csv("iris.csv")
iris


# In[59]:


import seaborn as sns
counts = iris["variety"].value_counts()
sns.barplot(data=counts)


# In[60]:


iris.info()


# In[27]:


iris["variety"].value_counts()


# In[61]:


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

# In[63]:


iris = iris.drop_duplicates(keep='first')


# In[64]:


iris[iris.duplicated]


# In[65]:


#reset the indexx
iris=iris.reset_index(drop=True)
iris


# In[66]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
iris.iloc[:,-1]=labelencoder.fit_transform(iris.iloc[:,-1])
iris.head()


# In[67]:


# Check the data types after label encoding
iris.info()


# #### Observation
# - The target column(variety) is still object type.It needs to to be converted to numeric(int)

# In[68]:


# Convert the target column data type to int
iris['variety'] = pd.to_numeric(labelencoder.fit_transform(iris['variety']))
print(iris.info())


# In[69]:


# Divide the dataset into x-columns and y- columns
X=iris.iloc[:,0:4]
Y=iris['variety']


# In[70]:


Y


# In[71]:


X


# In[76]:


#  further splig=tting of data into training and testing data sets
x_train, x_test,y_train, y_test = train_test_split(X,Y, test_size=0.3,random_state=1)
x_train


# In[78]:


#  further splig=tting of data into training and tesi=ting data sets
x_train, x_test,y_train, y_test = train_test_split(X,Y, test_size=0.3,random_state=2)
x_train


# In[79]:


#  further splig=tting of data into training and tesi=ting data sets
x_train, x_test,y_train, y_test = train_test_split(X,Y, test_size=0.3)
x_train.head(20)


# In[80]:


model = DecisionTreeClassifier(criterion = 'entropy',max_depth =None)
model.fit(x_train,y_train)


# In[81]:


#plot the decision tree
plt.figure(dpi=1200)
tree.plot_tree(model);


# In[82]:


fn=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
cn=['setosa', 'versicolor', 'virginica']
plt.figure(dpi=1200)
tree.plot_tree(model,feature_names = fn, class_names=cn,filled = True);


# In[83]:


# Predicting on test data
preds = model.predict(x_test)# predicting on test data set
preds


# In[85]:


print(classification_report(y_test,preds))


# ####
# - presision - presison and recall is applied to one single class precision = TP/TP+FP
# - recall - TP/TP+FN
# - f1Score - 2* p*R/P+R

# In[ ]:




