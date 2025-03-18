#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install xgboost


# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# In[3]:


df = pd.read_csv('diabetes.csv')
df


# In[5]:


# Split features and target
X = df.drop('class', axis=1)
y = df['class']

# Train test split(80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[8]:


# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(X_train_scaled)
print(X_test_scaled)


# In[22]:


# XGBoost classifier Instantiation with hyper paramater grid
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Hyperparameter grid
param_grid = {
    'n_estimators': [100,150,200,300],
    'learning_rate': [0.01,0.1,0.15],
    'max_depth': [2,3,4,5],
    'subsample': [0.8,1.0],
    'colsample_bytree': [0.8,1.0]
}

# Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# GridSearchCV with scoring =recall
grid_search = GridSearchCV(estimator=xgb,
                           param_grid=param_grid,
                           scoring='recall',
                           cv=skf,
                           verbose=1,
                           n_jobs=-1)


# In[23]:


grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validated Recall:", grid_search.best_score_)

y_pred = best_model.predict(X_test_scaled)


# In[24]:


# Evaluation
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassifiaction Report:\n", classification_report(y_test, y_pred))


# #### Identify feature importance scores using XGBClassifier

# In[25]:


best_model.feature_importances_


# In[30]:


features = pd.DataFrame(best_model.feature_importances_, index = df.iloc[:,:-1].columns, columns=["Importances"])
df1= features.sort_values(by = "Importances")


# In[32]:


df1


# In[31]:


import seaborn as sns
sns.barplot(data = df1, x= features.index, y= "Importances", hue = features.index,palette = "Set2")


# In[ ]:




