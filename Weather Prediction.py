#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datetime import datetime, timedelta
import time
from collections import namedtuple
import pandas as pd
import requests
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("C:\\Users\\Lenovo\\Downloads\\temperature.csv")
df.head(10)


# In[3]:


df.describe()


# In[4]:


df.corr()


# In[5]:


df.value_counts


# In[6]:


target_date = datetime(2016, 5, 16)
features = ["date", "meantempm", "meandewptm", "meanpressurem", "maxhumidity", "minhumidity", "maxtempm",
            "mintempm", "maxdewptm", "mindewptm", "maxpressurem", "minpressurem", "precipm"]
DailySummary = namedtuple("DailySummary", features)


# In[7]:


features


# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')

# Histogram of variable ApplicantIncome

df['Next_Tmax'].hist()

df['Next_Tmin'].hist()


# In[9]:


df.boxplot(column='Solar radiation')


# In[13]:


corr=df.corr()
import seaborn as sns 


# In[14]:


plt.figure(figsize=(20,15))
sns.heatmap(data=corr,annot=True)


# In[16]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
df.hist(bins=50, figsize=(20,30))
plt.savefig("attribute_histogram_plots")
plt.show()


# In[34]:


from matplotlib import style


# In[36]:


fig = plt.figure()
axl = plt.subplot2grid((1,1),(0,0))

style.use('ggplot')

sns.lineplot(x=df['Next_Tmax'], y=df['Next_Tmin'], data=df)
sns.set(rc={'figure.figsize':(50,40)})

plt.title("DailySummary")
plt.xlabel("Next_Tmax")
plt.ylabel("Next_Tmin ")
plt.grid(True)
plt.legend()


# In[38]:


fig = plt.figure()
axl = plt.subplot2grid((1,1),(0,0))

style.use('ggplot')

sns.lineplot(x=df['Present_Tmax'], y=df['Present_Tmin'], data=df)
sns.set(rc={'figure.figsize':(50,40)})

plt.title("DailySummary")
plt.xlabel("Present_Tmax")
plt.ylabel("Present_Tmin ")
plt.grid(True)
plt.legend()


# In[ ]:





# In[ ]:




