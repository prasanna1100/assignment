#!/usr/bin/env python
# coding: utf-8

# ### Assumptions in Multilinear Regression
# - Linearity: The relationship between the predictord and the response is linear
# - Independence: Observations are independent of each other.
# - Homoscedasticity: The residuals (difference between observed and predicted values)exihibit constant variance at all levels of the predictor
# - Normal Distribution of Errors: The residuals of the model are normally distributed.
# - No Multicollinearity:The independent variables should not be too highly correlated with each other Violations of these assumptions may lead to in efficeincy in the regression parameters and unreliable predictors

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# In[4]:


# Read the data from csv file
cars = pd.read_csv("Cars.csv")
cars.head()


# In[6]:


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

# In[7]:


cars.info()


# In[8]:


# Check for missing values
cars.isna().sum()


# ### Observations
# - No missing values are observed.
# - There are 81 observations
# - The data types of each column is relevant and valid with respect to the columns.

# In[10]:


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


# In[ ]:




