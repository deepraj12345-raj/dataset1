#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[7]:


churn = pd.read_csv("C:\\Users\\Lenovo\\Downloads\\Telecom_customer_churn.csv")


# In[8]:


churn


# In[9]:


print(churn.shape)


# In[10]:


print(len(churn.customerID.unique()))


# In[11]:


columns = ['customerID','gender','SeniorCitizen','Partner','Dependents','tenure','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod','MonthlyCharges','TotalCharges','Churn'
]


# In[12]:


churn.shape


# In[13]:


churn.value_counts


# In[14]:


churn.head()


# In[19]:


n = churn.pivot_table(index="customerID", columns="PhoneService")
n2 = churn.pivot_table(index="customerID", columns="InternetService")
n3 = churn.pivot_table(index="customerID", columns="MultipleLines")
n4 = churn.pivot_table(index="customerID", columns="TotalCharges")
n5 = churn.pivot_table(index="customerID", columns="MonthlyCharges")
n6 = churn.pivot_table(index="customerID", columns="Churn")


# In[20]:


n.head()


# In[21]:


ser = np.matrix(n)


# In[22]:


ser = pd.DataFrame(ser,columns=n.columns)
ser.head()


# In[23]:


final = pd.concat([ser],1) 
final.head() # view


# In[24]:


print(final.isnull().sum())
print('\n Total:\t\t', final.isnull().any(1).sum())


# In[25]:


final[final.isnull().any(1)]


# In[26]:


final = final.dropna()


# In[27]:


final.to_csv('f.csv',index=False)
final = pd.read_csv('f.csv')


# In[28]:


final.head()


# In[30]:


final.to_csv('fin.csv')# write data


# In[40]:


final.head()


# In[41]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[42]:


# Plots for categorical attributes
import matplotlib.pyplot as plt
churn.hist(bins=70, figsize=(10, 20))
plt.show()


# In[ ]:


plt.figure()
final.MonthlyCharges.hist(grid = False)
plt.xlabel('MonthlyCharges')
plt.ylabel('MonthlyCharges')
plt.savefig('base_hist.png')


# In[66]:


import seaborn as sns 
correlation = churn.corr()
plt.figure(figsize=(14, 8))
sns.heatmap(correlation, annot=True, linewidths=0, vmin=-1, cmap="RdBu_r")


# In[68]:


sns.displot(churn["customerID"])
plt.title("churn")

