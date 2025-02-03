#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[2]:


data1 = pd.read_csv("NewspaperData.csv")
data1


# In[3]:


data1.info()


# In[4]:


data1.describe()


# In[5]:


plt.scatter(data1["daily"], data1["sunday"])


# In[6]:


data1["daily"].corr(data1["sunday"])


# In[7]:


# Build Regression model

import statsmodels.formula.api as smf
model = smf.ols("sunday~daily",data = data1).fit()


# In[8]:


model.summary()


# In[9]:


x = data1["daily"].values
y = data1["sunday"].values
plt.scatter(x, y, color = "m", marker = "o", s = 30)
b0 = 13.84
b1 = 1.33

y_hat = b0 + b1*x

plt.plot(x, y_hat, color = "g")

plt.xlabel('x')
plt.ylabel('y')
plt.show()


# Observations
# There are no missing values
# the daily column values appears to be right skewed
# The sunday column values also appear to be right skewed
# There are two outliers in both daily colun and also in sunday column observed from the bxplot

# **Scatter plot and Corelation Strength

# In[10]:


x= data1["daily"]
y= data1["sunday"]
plt.scatter(data1["daily"], data1["sunday"])
plt.xlim(0, max(x) + 100)
plt.ylim(0, max(y) + 100)
plt.show()


# In[11]:


data1.corr(numeric_only=True)


# **Observations on correlation strength
# The relation between x(daily) and y(sunday) is seen to be linera as seen from scatter plot
# the corr is strong and positive with Pearson's corr coeff of 0.958154

# In[12]:


import statsmodels.formula.api as smf
model1 = smf.ols("sunday~daily",data = data1).fit()


# In[13]:


model.summary()


# Prediction eqn is beta_0 = 13.8356 beta_1=13397x
# beta0+beta1*x = y_hat

# The probability (p-value) for intercept (beta_0) is 0.05
# Therefore the intercept coefficient many not be that much significant in prediction
# However the p+value for "daily" (beta_1)is 0.00<0.005
# Therefore the beta_! coefficient is highly significant and is contributint to prediction

# In[14]:


# plot the linear regression line using seaborn regplot() method
sns.regplot(x="daily", y="sunday", data=data1)
plt.xlim([0,1250])
plt.show()


# Fit a Linear Regression Model

# In[15]:


# Build regression model

import statsmodels.formula.api as smf
model1 = smf.ols("sunday~daily",data = data1).fit()


# In[16]:


model1.summary()


# In[17]:


# print the fitted line coefficients (Beta-0 and Beta-1)
model1.params


# In[21]:


# print the quality of fitted line(R squared values)
(model1.rsquared,model1.rsquared_adj)


# In[22]:


### predict on new Data


# In[24]:


# predict for 200 and 300 daily circulation
newdata=pd.Series([200,300,1500])


# In[26]:


data_pred=pd.DataFrame(newdata,columns=['daily'])
data_pred


# In[27]:


model1.predict(data_pred)


# In[29]:


# predict on given all training data

pred = model1.predict(data1["daily"])
pred


# In[30]:


# Add predicted values as a column in data1
data1["Y_hat"] = pred
data1


# In[32]:


# Compute the error values(residuals) and as another column
data1["residuals"]= data1["sunday"]-data1["Y_hat"]
data1


# In[35]:


# Compute Mean Squared Error for the model

mse = np.mean((data1["daily"]-data1["Y_hat"])**2)
rmse = np.sqrt(mse)
print("MSE: ",mse)
print("RMSE: ",rmse)


# ### Checking the model residuals scatter plot

# In[36]:


plt.scatter(data1["Y_hat"], data1["residuals"])


# #### Observation:
# - There appears to be no trend and the residuals are randomly placed around the zero error line
# - Hence the assumption of homoscedasticty is satisfied (constant variance in residuals)

# In[37]:


#plot the Q-Q plot(to check the normality of residuals)
import statsmodels.api as sm
sm.qqplot(data1["residuals"], line='45', fit=True)
plt.show()


# In[39]:


sns.histplot(data1["residuals"], kde =True)


# #### Observations:
# - The data points are seen to closely follow the reference line of normality
# - Hence the residuals are approximately normally distributed as also can be seen from the data

# In[ ]:




