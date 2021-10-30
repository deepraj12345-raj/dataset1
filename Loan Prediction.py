#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder


# In[2]:


df = pd.read_csv("C:\\Users\\Lenovo\\Downloads\\loan_prediction.csv")
df.head(10)


# In[3]:


df.describe()


# In[5]:


df['Property_Area'].value_counts()
df['Property_Area'].value_counts()


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')

# Histogram of variable ApplicantIncome

df['ApplicantIncome'].hist()


# In[7]:


df.boxplot(column='ApplicantIncome')


# In[10]:


df.boxplot(column='ApplicantIncome', by = 'Education')


# In[11]:


df['LoanAmount'].hist(bins=50)


# In[12]:


df.boxplot(column='LoanAmount', by = 'Gender')


# In[13]:


loan_approval = df['Loan_Status'].value_counts()['Y']
print(loan_approval)


# In[14]:


pd.crosstab(df ['Credit_History'], df ['Loan_Status'], margins=True)


# In[15]:


def percentageConvert(ser):
    return ser/float(ser[-1])


# In[16]:


df.head()


# In[17]:


df['Self_Employed'].fillna('No',inplace=True)


# In[18]:


df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']

# Looking at the distribtion of TotalIncome
df['LoanAmount'].hist(bins=20)


# In[19]:


df['LoanAmount_log'] = np.log(df['LoanAmount'])

# Looking at the distribtion of TotalIncome_log
df['LoanAmount_log'].hist(bins=20)


# In[20]:


df['Gender'].fillna(df['Gender'].mode()[0],inplace=True)

# Impute missing values for Married
df['Married'].fillna(df['Married'].mode()[0],inplace=True)

# Impute missing values for Dependents
df['Dependents'].fillna(df['Dependents'].mode()[0],inplace=True)

# Impute missing values for Credit_History
df['Credit_History'].fillna(df['Credit_History'].mode()[0],inplace=True)

# Convert all non-numeric values to number
cat=['Gender','Married','Dependents','Education','Self_Employed','Credit_History','Property_Area']

for var in cat:
    le = preprocessing.LabelEncoder()
    df[var]=le.fit_transform(df[var].astype('str'))
df.dtypes


# In[24]:


ID_col = ['Loan_ID']
target_col = ["Loan_Status"]
cat_cols = ['Credit_History','Dependents','Gender','Married','Education','Property_Area','Self_Employed']


# In[26]:


df.isnull().sum()


# In[27]:


df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
df['LoanAmount_log'].fillna(df['LoanAmount_log'].mean(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean(), inplace=True)
df['ApplicantIncome'].fillna(df['ApplicantIncome'].mean(), inplace=True)
df['CoapplicantIncome'].fillna(df['CoapplicantIncome'].mean(), inplace=True)

#Imputing Missing values with mode for categorical variables
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)


# In[28]:


df['TotalIncome']=df['ApplicantIncome'] + df['CoapplicantIncome']

df['TotalIncome_log'] = np.log(df['TotalIncome'])

#Histogram for Total Income
df['TotalIncome_log'].hist(bins=20)


# In[ ]:




