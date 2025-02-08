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


# In[4]:


# Read the data from csv file
cars = pd.read_csv("Cars.csv")
cars.head()


# In[5]:


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

# In[6]:


cars.info()


# In[7]:


# Check for missing values
cars.isna().sum()


# ### Observations
# - No missing values are observed.
# - There are 81 observations
# - The data types of each column is relevant and valid with respect to the columns.

# In[8]:


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


# In[15]:


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


# In[16]:


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


# In[17]:


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


# In[9]:


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

# In[10]:


cars[cars.duplicated()]


# In[11]:


sns.set_style(style='darkgrid')
sns.pairplot(cars)


# In[12]:


cars.corr()


# #### Observation
# - Between x and y all the variables are showing moderate to high correlation strengths, highest being between HP and MPG
# - Therefore this dataset qualifies for building a multiple regression model to predict MPG
# - Among x columns (x1,x2,x3 andx4), some very high correlation strengths are observed between SP vsHP, VOL vs WT
# - The High correlation among x columns is not desirable as it might lead to multicollineraity problem

# #### Preparing a preliminary model considering all X columns

# In[13]:


# Build model
# import statsmodels.formula.api as smf
model1 = smf.ols('MPG~WT+VOL+SP+HP', data=cars).fit()


# In[15]:


model1.summary()


#  #### Observations from model summary
#  - The R-Squared and adjusted R-Squared values are good and about 75% of variabilty in Y is explained by X columns
#  - The probability value with respect to F-statistic is close to zer, indictaing thata ll or someof X coluns are significant
#  - The p-values for VOL and WT are higher than 5% indicating some interaction issue among themselves , which need to be further explored

# #### Performance metrics for model1

# In[18]:


# Find the performance metrics
# Create a data frame with actual y and predicted y columns

df1 = pd.DataFrame()
df1["actual_y1"] = cars["MPG"]
df1.head()


# In[19]:


# predict for the given x columns

pred_y1 = model1.predict(cars.iloc[:,0:4])
df1["pred_y1"] = pred_y1
df1.head()


# In[26]:


# Compute the Mean Squared Error(MSE) for model

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df1["actual_y1"], df1["pred_y1"])
print("MSE :", mse)
print("RMSE :",np.sqrt(mse))


# #### Checking for multicollinearity among X-Columns using VIF method

# In[27]:


cars.head()


# In[25]:


# Compute VIF values
rsq_hp = smf.ols('HP~WT+VOL+SP',data=cars).fit().rsquared
vif_hp = 1/(1-rsq_hp)

rsq_wt = smf.ols('WT~HP+VOL+SP',data=cars).fit().rsquared  
vif_wt = 1/(1-rsq_wt) 

rsq_vol = smf.ols('VOL~WT+SP+HP',data=cars).fit().rsquared  
vif_vol = 1/(1-rsq_vol) 

rsq_sp = smf.ols('SP~WT+VOL+HP',data=cars).fit().rsquared  
vif_sp = 1/(1-rsq_sp) 

# Storing vif values in a data frame
d1 = {'Variables':['Hp','WT','VOL','SP'],'VIF':[vif_hp,vif_wt,vif_vol,vif_sp]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame


# #### Observations
# - The ideal range of VIF values shall be between 0 to 10. However slighty higher values can be tolerated.
# - As seen from the very high VIF values for VOL and WT ,it is clear that they are prone to multicollineraity problem.
# - Hence it is declared to drop one of the column (either VOL or WT)to overcome the multicollinearity.
# - It is decieded to drop WT and retain VOL column in futher models. 

# In[26]:


cars1 = cars.drop("WT", axis=1)
cars1.head()


# In[27]:


# Build model2 on cars dataset
import statsmodels.formula.api as smf
model2 = smf.ols('MPG~VOL+SP+HP',data=cars1).fit()


# In[28]:


model2.summary()


# #### Performance metrics for model2

# In[29]:


# Find the performance metrics
# Create a data frame with actual y and predicted y 
df2 = pd.DataFrame()
df2["actual_y2"] = cars["MPG"]
df2.head()


# In[30]:


# Predict for the given x data columns

pred_y2 = model2.predict(cars1.iloc[:,0:4])
df2["pred_y2"] = pred_y2
df2.head()


# In[31]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df2["actual_y2"], df2["pred_y2"])
print("MSE :", mse)
print("RMSE :",np.sqrt(mse))


# #### Observations 
# - The adjusted R-Squared value improved slighty to 0.76
# - All the p-values for model parameters are less than 5% they are significant
# - Therefore the HP, VOL, SP columns are finalized as the significant predictor for the MPG
# - There is no improvement in MSE value

# #### IDentification of High Influence points(spatial outliers)

# In[32]:


cars1.shape


# In[33]:


# Define variables and assign values
k = 3 # no.of X-columns
n = 81 # no.of observations(rows)
levarage_cutoff = 3*((k + 1)/n)
levarage_cutoff


# In[34]:


from statsmodels.graphics.regressionplots import influence_plot
influence_plot(model1,alpha=0.5)

y=[i for i in range(-2,8)]
x=[levarage_cutoff for i in range(10)]
plt.plot(x,y,'r+')

plt.show()


# - from the above plot it is evident that data points 65,70,76,78,79,80 are the influencers.
# - as their H Leverage values are higher and size is higher

# In[35]:


cars1[cars1.index.isin([65,70,76,78,79,80])]


# In[36]:


#Discard the data points which are influencers and reassign the row number
cars2=cars1.drop(cars1.index[[65,70,76,78,79,80]],axis=0).reset_index(drop=True)


# In[37]:


cars2


# #### Build Model3 on cars2 dataset

# In[38]:


# rebuild the model model
model3 = smf.ols('MPG~VOL+SP+HP',data=cars1).fit()
model3.summary()


# #### Perormance Metrics for model3

# In[40]:


df3= pd.DataFrame()
df3["actual_y3"] =cars2["MPG"]
df3.head()


# In[41]:


# Predict on all X data Columns
pred_y3 = model3.predict(cars2.iloc[:,0:3])
df3["pred_y3"] = pred_y3
df3.head()


# In[46]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df3["actual_y3"], df3["pred_y3"])
print("MSE :", mse)
print("RMSE :",np.sqrt(mse))


# #### comparison of models
# | Metric            | model1  | model2  | model3   |
# |-------------------|---------|---------|----------|
# |R-squared          | 0.771   | 0.770   | 0.885    |
# |Adj.R -squared     | 0.758   | 0.761   | 0.880    |
# |MSE                | 18.89   | 18.91   | 8.68     |
# |RSME               |4.34     | 4.34    | 2.94     |

# #### Check the validity of model assumptions for model3

# In[47]:


model3.resid
model3.fittedvalues


# In[48]:


# The Model is built with VOL, SP, HP by ignoring WT
import statsmodels.api as sm
qqplot=sm.qqplot(model3.resid,line='q')
plt.title("Normal q-q plot of residuals")
plt.show()


# In[50]:


sns.displot(model3.resid, kde = True)


# In[51]:


def get_standardized_values( vals ):
    return (vals - vals.mean())/vals.std()


# In[52]:


plt.figure(figsize=(6,4))
plt.scatter(get_standardized_values(model3.fittedvalues),
            get_standardized_values(model3.resid))

plt.title('Residual Plot')
plt.xlabel('Standardized Fitted values')
plt.ylabel('Standardized residual values')
plt.show()


# In[ ]:




