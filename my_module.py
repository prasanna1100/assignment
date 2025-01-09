#!/usr/bin/env python
# coding: utf-8

# In[1]:


greet = lambda name : print(f"Good morning {name}!")
greet("Sowmya")


# In[2]:


# Product of three numbers
product = lambda a,b,c : a*b*c


# In[3]:


product(2,3,4)


# In[4]:


# Lambda functions with List Comprehension
even = lambda L : [x for x in L if x%2 ==0]


# In[5]:


my_list = [100,3,9,38,43,48,56]
even(my_list)


# In[6]:


# Lambda functions with list Comprehnsion
odd = lambda L : [x for x in L if x%2 !=0]


# In[7]:


my_list = [100,3,9,38,43,48,56]
odd(my_list)


# In[ ]:




