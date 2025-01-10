#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Import numpy as np
import numpy as np


# In[3]:


# Create 1D numpy array
x = np.array([45,67,57,60])
print(x)
print(type(x))
print(x.dtype)


# In[4]:


# verify the data type in array
x = np.array([45,68,58,9.8])
print(x)
print(type(x))
print(x.dtype)


# In[5]:


# verify the data type
x = np.array(["A",45,68,58,9.8])
print(x)
print(type(x))
print(x.dtype)


# In[7]:


# Create a 2D array
a2 = np.array([[20,40],[30,60]])
print(a2)
print(type(a2))
print(a2.shape)


# In[8]:


# Reshaping an array
a = np.array([10,20,30,40])
b = a.reshape(2,2)
print(b)
print(b.shape)


# In[13]:


# Create an array with arange()
c = np.arange(3,10)
print(c)
type(c)


# In[14]:


# USe of around()
d = np.array([1.345, 2.654, 8.868])
print(d)
np.around(d,2)


# In[18]:


# Use of np.sqrt()
d = np.array([1.345, 2.654, 8.868])
print(d)
print(np.around(np.sqrt(d),2))


# In[19]:


# Create a 2D array
a1 = np.array([[3,4,5,8],[7,2,8,np.NAN]])
print(a1)
print(a1.shape)
print(a1.dtype)


# In[26]:


# Use of astype() to convert the data type
a1_copy1 = a1.astype(str)
print(a1_copy1)
a1_copy1.dtype


# In[28]:


# Mathematical Operations on rows and columns
a2 =np.array([[3,4,6],[7,9,10],[4,6,12]])
a2


# In[29]:


print(a2.sum(axis = 1))
print(a2.sum(axis = 0))


# In[31]:


# Find mean values of rows and colms
print(a2)
print(a2.mean(axis = 0))
print(a2.mean(axis = 1))


# In[32]:


# Matrix Operations
a3 = np.array([[3,4,5],[7,2,8],[9,1,6]])
print(a3)
np.fill_diagonal(a3,0)
print(a3)


# In[36]:


# Define two matrices and multiply them
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Perform matrix multiplication
c = np.matmul(A, B)
print(c)


# In[37]:


# Print the Transpose of the matrix
print(A.T)
print(B.T)


# In[38]:


# Accessing the array elements
a4 = np.array(([3,4,5],[7,2,8],[9,8,7],[1,2,3]))
a4


# In[41]:


print(a4.sum(axis = 0))
print(a4.sum(axis = 1))


# In[ ]:




