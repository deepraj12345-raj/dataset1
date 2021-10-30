#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np 

# data processing
import pandas as pd 

# data visualization
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
from matplotlib import style

# Algorithms
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB


# In[48]:


housing  = pd.read_csv("C:\\Users\\Lenovo\\Project-Housing_splitted\\test.csv")
housing  = pd.read_csv("C:\\Users\\Lenovo\\Project-Housing_splitted\\train.csv")


# In[49]:


housing .info()
housing .info()


# In[28]:


test.describe()
train.describe()


# In[50]:


print(housing .shape)
print(housing .shape)


# In[51]:


print(len(housing .Id.unique()))
print(len(housing .Id.unique()))


# In[52]:


housing .head()
housing .head()


# In[53]:


import matplotlib.pyplot as plt
housing.hist(bins=70, figsize=(30, 40))
plt.show()


# In[54]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


# In[21]:


import numpy as np
housing['SalePrice'] = pd.cut(housing['YrSold'], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])
housing['SalePrice'].hist()
plt.show()


# In[22]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["MoSold"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
print(strat_test_set['YrSold'].value_counts() / len(strat_test_set))


# In[23]:


housing.plot(kind='scatter', x='LotFrontage', y='LotArea', alpha=0.4, s=housing['MSSubClass']/100, label='MSSubClass',
figsize=(12, 8), c='SalePrice', cmap=plt.get_cmap('jet'), colorbar=True)
plt.legend()
plt.show()


# In[56]:


housing["SaleCondition"] = np.ceil(housing["SalePrice"]/ 1.5)
housing["SaleCondition"].where(housing["SaleCondition"] < 5, 5.0, inplace=True)


# In[57]:


from sklearn.model_selection import StratifiedShuffleSplit


# In[58]:


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)


# In[60]:


for train_index, test_index in split.split(housing, housing['SaleCondition']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[75]:


housing = strat_train_set.copy()
housing.plot(kind="scatter", x="SaleCondition", y="SaleCondition", alpha=0.4,
            s=housing["YrSold"]/50, label="MSSubClass",
            c="SalePrice", cmap=plt.get_cmap("jet"), colorbar=True,
            figsize=(7,7))
plt.legend()


# In[79]:


from pandas.plotting import scatter_matrix
attributes = ['SalePrice', 'SaleCondition',
             'YearRemodAdd', 'YearBuilt']
scatter_matrix(housing[attributes], figsize=(12,8))


# In[81]:


housing.plot(kind='scatter', x='SalePrice', y='YearBuilt',
            alpha=0.1, figsize=(8,5))


# In[82]:


housing.head(3)


# In[86]:


corr_matrix = housing.corr()
corr_matrix['SalePrice'].sort_values(ascending=False)


# In[89]:


housing = strat_train_set.drop("SalePrice", axis=1)
housing_labels = strat_train_set['SalePrice'].copy()


# In[179]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="SalePrice")
housing_num = housing.drop('YrSold', axis=1)
housing_num.head()


# In[182]:


correlation = housing.corr()
plt.figure(figsize=(70, 60))
sns.heatmap(correlation, annot=True, linewidths=0, vmin=-1, cmap="RdBu_r")


# In[ ]:




