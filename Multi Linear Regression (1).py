#!/usr/bin/env python
# coding: utf-8

# ### Assumptions in Multilinear Regression
# - Linearity: The relationship between the predictord and the response is linear
# - Independence: Observations are independent of each other.
# - Homoscedasticity: The residuals (difference between observed and predicted values)exihibit constant variance at all levels of the predictor
# - Normal Distribution of Errors: The residuals of the model are normally distributed.
# - No Multicollinearity:The independent variables should not be too highly correlated with each other Violations of these assumptions may lead to in efficeincy in the regression parameters and unreliable predictors

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# In[2]:


# Read the data from csv file
cars = pd.read_csv("Cars.csv")
cars.head()


# In[3]:


# Rearrange the columns 
cars = pd.DataFrame(cars, columns=["HP", "VOL", "SP", "WT", "MPG"])
cars.head()


# ### Description of columns
# - MPG : Milege of the car (Mile per gallon)
# - HP  : Horse Power of the car
# - VOL : Volume of the car (Sixe of the car)
# - SP  : Top speed of the car (Miles per hour)
# - WT  : Weight of the car (Pounds)

# ### EDA

# In[4]:


cars.info()


# In[5]:


# Check for missing values
cars.isna().sum()


# ### Observations
# - No missing values are observed.
# - There are 81 observations
# - The data types of each column is relevant and valid with respect to the columns.

# In[6]:


# Create a figure with two subplots (one above the other)
fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

# creating a boxplot
sns.boxplot(data=cars, x='HP', ax=ax_box, orient='h')
ax_box.set(xlabel='') 

# Creating a histogram in the same x-axis
sns.histplot(data=cars, x='HP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

#Adjust layout
plt.tight_layout()
plt.show()


# In[7]:


# Create a figure with two subplots (one above the other)
fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

# creating a boxplot
sns.boxplot(data=cars, x='VOL', ax=ax_box, orient='h')
ax_box.set(xlabel='') 

# Creating a histogram in the same x-axis
sns.histplot(data=cars, x='VOL', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

#Adjust layout
plt.tight_layout()
plt.show()


# In[8]:


# Create a figure with two subplots (one above the other)
fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

# creating a boxplot
sns.boxplot(data=cars, x='SP', ax=ax_box, orient='h')
ax_box.set(xlabel='') 

# Creating a histogram in the same x-axis
sns.histplot(data=cars, x='SP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

#Adjust layout
plt.tight_layout()
plt.show()


# In[9]:


# Create a figure with two subplots (one above the other)
fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

# creating a boxplot
sns.boxplot(data=cars, x='WT', ax=ax_box, orient='h')
ax_box.set(xlabel='') 

# Creating a histogram in the same x-axis
sns.histplot(data=cars, x='WT', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

#Adjust layout
plt.tight_layout()
plt.show()


# In[10]:


# Create a figure with two subplots (one above the other)
fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

# creating a boxplot
sns.boxplot(data=cars, x='MPG', ax=ax_box, orient='h')
ax_box.set(xlabel='') 

# Creating a histogram in the same x-axis
sns.histplot(data=cars, x='MPG', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

#Adjust layout
plt.tight_layout()
plt.show()


# #### Observations from boxplot and histograms
# - There are some extreme values(outliers) observed in towards the right tail of SP and HP distributions
# - In VOL and WT columns , a few outliers are observed in both tails of their distributions
# - The extreme values of cars data may have come from the specially designed nature of cars
# - As this is multi-dimensional data, the outliers with respect to spatial dimensions may have to be considered while building the regression model

# In[12]:


cars[cars.duplicated()]


# In[13]:


sns.set_style(style='darkgrid')
sns.pairplot(cars)


# In[14]:


cars.corr()


# #### Observation
# - Between x and y all the variables are showing moderate to high correlation strengths, highest being between HP and MPG
# - Therefore this dataset qualifies for building a multiple regression model to predict MPG
# - Among x columns (x1,x2,x3 andx4), some very high correlation strengths are observed between SP vsHP, VOL vs WT
# - The High correlation among x columns is not desirable as it might lead to multicollineraity problem

# #### Preparing a preliminary model considering all X columns

# In[19]:


# Build model
# import statsmodels.formula.api as smf
model1 = smf.ols('MPG~WT+VOL+SP+HP', data=cars).fit()


# In[20]:


model1.summary()


# In[ ]:




