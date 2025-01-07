#!/usr/bin/env python
# coding: utf-8

# In[ ]:


Sets
Unordered
Mutable
unique values
Allows immutable elements
no indexing or slicing


# s1 ={1,8,9,0,10,9,0,1,9}
# print(s1)

# In[1]:


s1 ={1,8,9,0,10,9,0,1,9}
print(s1)


# In[5]:


## Union operation using | operator
s1 = {1,2,3,4}
s2 = {3,4,5,6}
s1 | s2


# In[6]:


s1.union(s2)


# In[7]:


## intersection using & operator
s1 = {1,2,3}
s2 = {3,4,5}
s1 & s2


# In[9]:


s1.intersection(s2)


# In[10]:


# diference of two sets 
s1 = {2,3,5,6,7}
s2 = {5,6,7}
s1 - s2


# In[11]:


s2 - s1


# In[12]:


s1.difference(s2)


# In[13]:


# Symmertic _difference
s1 = {1,2,3,4,5}
s2 = {4,5,6,7,8}
s1.symmetric_difference(s2)


# In[14]:


s1 = {1,2,3,4,5}
s2 = {1,2,3}
s2.issubset(s1)


# In[15]:


s1.issubset(s2)


# In[16]:


s2.issuperset(s1)


# In[17]:


s1.issuperset(s2)


# In[ ]:


#STRINGS :
collection of alpha numeric characters 
Strings are immutable
indexing
iterable Objects


# In[18]:


str1 = "Welcome aiml class"
str2 = 'We started with python'
str3 = '''This an awesome class'''


# In[19]:


print(type (str1))
print(type (str2))
print(type (str3))


# In[20]:


# Slicing in strings
print(str1)
str1[5:10]


# In[21]:


dir(str)


# In[22]:


# Split 
print(str1)
str1.split()


# In[ ]:


splits into tokens it will poduce list of word tokens


# In[24]:


# join()
str4 = "Hello. How are you?"
' '.join(str4)


# In[27]:


# Ue of strip() method(to remove the empty spaces)
str5= "     Hello, How are you?    "
str5


# In[26]:


str5.strip()


# In[29]:


# industry usecase of data sales analysis Example-1
# Example dictionary for product sales analysis
sales_data = {
    "ProductID": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    "ProductName": ["Laptop", "Mouse", "Keyboard", "Monitor", "Chair", "Desk", "Webcam", "Headphones", "Printer", "Tablet"],
    "Category": ["Electronics", "Accessories", "Accessories", "Electronics", "Furniture", "Furniture", "Electronics", "Accessories", "Electronics", "Electronics"],
    "PriceRange": ["High", "Low", "Low", "Medium", "Medium", "Medium", "Low", "Low", "Medium", "High"],
    "StockAvailable": [15, 100, 75, 20, 10, 8, 50, 60, 25, 12],
}
for k,v in sales_data.items():
    print(k,set(v), end =',')
    print('/n')


# In[30]:


# industry usecase data on review analysis Example-1
# Original reviews dictionary
reviews = {
    "Review1": "The product quality is excellent and delivery was prompt. The product functionality is versatile",
    "Review2": "Good service but the packaging could have been better. The customer service has to improve",
    "Review3": "The product works fine, but the customer support is not very helpful. I rate the product as excellent",
}

# Result dictionary to store analysis of reviews
review_analysis = {}

# Process each review
for key, review in reviews.items():
    # Split the review into words
    words = review.lower().replace('.', '').replace(',', '').split()
    # Create a sub-dictionary with word count and unique words
    review_analysis[key] = {
        "WordCount": len(words),
        "UniqueWords": list(set(words))
    }

review_analysis


# In[ ]:




