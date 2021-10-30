#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[5]:


teams_df = pd.read_csv("C:\\Users\\Lenovo\\Downloads\\baseball.csv")


# In[6]:


print(teams_df)


# In[49]:


teams_df.describe()


# In[7]:


cols = ['yearID','lgID','teamID','franchID','divID','Rank','G','Ghome','W','L','DivWin','WCWin','LgWin','WSWin','R','AB','H','2B','3B','HR','BB','SO','SB','CS','HBP','SF','RA','ER','ERA','CG','SHO','SV','IPouts','HA','HRA','BBA','SOA','E','DP','FP','name','park','attendance','BPF','PPF','teamIDBR','teamIDlahman45','teamIDretro','franchID','franchName','active','NAassoc']


# In[35]:


teams_df.columns  


# In[36]:


print(len(teams_df))


# In[44]:


#Dropping your unnecesary column variables.
drop_cols = ['lgID','franchID','divID','Rank','Ghome','L','DivWin','WCWin','LgWin','WSWin','SF','name','park','attendance','BPF','PPF','teamIDBR','teamIDlahman45','teamIDretro','franchID','franchName','active','NAassoc']


# In[45]:


print(teams_df.sum(axis=0).tolist())


# In[50]:


import seaborn as sb
sb.set_context("notebook", font_scale=2.5)

from matplotlib import pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[51]:


sb.pairplot(power,size=3)


# WINNING MATCH

# In[52]:


sns.displot(teams_df["W"])
plt.title("WINNING MATCH")


# In[ ]:




