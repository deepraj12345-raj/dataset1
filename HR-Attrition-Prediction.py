#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


# In[3]:


data = pd.read_csv("C:\\Users\\Lenovo\\WA_Fn-UseC_-HR-Employee-Attrition.csv")
data.head()


# In[4]:


data.isnull().sum()


# In[5]:


data.dtypes


# In[41]:


num_col=[]
for i in data.columns:
    if(data[i].dtypes!=object and data[i].nunique()<30):
        print(i, data[i].unique())
        num_col.append(i)
#print(num_col)


# In[42]:


f,ax=plt.subplots(figsize=(20,20))
sns.heatmap(data.corr(),annot=True,linewidth=.5,fmt='.1f')


# In[46]:


data.describe().transpose()


# In[48]:


data.isnull().any().any()


# In[68]:


y = data.iloc[:, 1]
y.head()


# In[70]:


data.columns


# In[72]:


X = data.loc[:,['DailyRate', 'DistanceFromHome', 'EnvironmentSatisfaction',
       'HourlyRate', 'JobInvolvement', 'JobSatisfaction',
       'RelationshipSatisfaction', 'StockOptionLevel',
       'TrainingTimesLastYear']]
X.head()


# In[75]:


from sklearn.ensemble import ExtraTreesRegressor
selection = ExtraTreesRegressor()
selection.fit(X,y)


# In[76]:


print(selection.feature_importances_)


# In[77]:


plt.figure(figsize = (12,8))
feat_importances = pd.Series(selection.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()

