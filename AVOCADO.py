#!/usr/bin/env python
# coding: utf-8

# In[43]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[44]:


avocado_df = pd.read_csv("C:\\Users\\Lenovo\\avocado.csv")
avocado_df.head()


# In[45]:


avocado_df.tail(10)


# In[46]:


avocado_df = avocado_df.sort_values("Date")


# In[47]:


plt.figure(figsize=(10,10))
plt.plot(avocado_df['Date'], avocado_df['AveragePrice'])


# In[48]:


avocado_df


# In[49]:


plt.figure(figsize=[25,12])
sns.countplot(x = 'region', data = avocado_df)
plt.xticks(rotation = 45)


# In[50]:


plt.figure(figsize=[25,12])
sns.countplot(x = 'year', data = avocado_df)
plt.xticks(rotation = 45)


# In[51]:


avocado_prophet_df = avocado_df[['Date', 'AveragePrice']]
avocado_prophet_df


# In[52]:


avocado_prophet_df = avocado_prophet_df.rename(columns={'Date':'ds', 'AveragePrice':'y'})
avocado_prophet_df


# # PART 2

# In[53]:


avocado_df


# In[54]:


avocado_df_sample = avocado_df[avocado_df['region']=='West']
avocado_df_sample


# In[55]:


avocado_df_sample


# In[56]:


avocado_df_sample = avocado_df_sample.sort_values("Date")
plt.figure(figsize=(10,10))
plt.plot(avocado_df_sample['Date'], avocado_df_sample['AveragePrice'])


# In[57]:


avocado_df_sample = avocado_df_sample.rename(columns={'Date':'ds', 'AveragePrice':'y'})


# In[58]:


correlation = avocado_df.corr()
plt.figure(figsize=(14, 8))
sns.heatmap(correlation, annot=True, linewidths=0, vmin=-1, cmap="RdBu_r")


# In[59]:


sns.displot(avocado_df["AveragePrice"])
sns.displot(avocado_df["year"])
plt.title("Total Volume")


# In[ ]:




