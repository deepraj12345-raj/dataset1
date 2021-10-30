#!/usr/bin/env python
# coding: utf-8

# In[66]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[120]:


train_data = pd.read_excel("C:\\Users\\Lenovo\\Flight_Ticket_Participant_Datasets\\Test_set.xlsx")
test_data = pd.read_excel("C:\\Users\\Lenovo\\Flight_Ticket_Participant_Datasets\\Data_Train.xlsx")


# In[121]:


test_data.head()


# In[69]:


train_data.info()


# In[70]:


train_data["Duration"].value_counts()


# In[71]:


train_data.dropna(inplace = True)


# In[72]:


train_data.isnull().sum()


# In[73]:


train_data["Journey_day"] = pd.to_datetime(train_data.Date_of_Journey, format="%d/%m/%Y").dt.day


# In[74]:


train_data["Journey_month"] = pd.to_datetime(train_data["Date_of_Journey"], format = "%d/%m/%Y").dt.month


# In[75]:


train_data.head()


# In[76]:


train_data.drop(["Date_of_Journey"], axis = 1, inplace = True)


# In[77]:


# Extracting Hours
train_data["Dep_hour"] = pd.to_datetime(train_data["Dep_Time"]).dt.hour

# Extracting Minutes
train_data["Dep_min"] = pd.to_datetime(train_data["Dep_Time"]).dt.minute

# Now we can drop Dep_Time as it is of no use
train_data.drop(["Dep_Time"], axis = 1, inplace = True)


# In[78]:


train_data.head()


# In[79]:


train_data["Arrival_hour"] = pd.to_datetime(train_data.Arrival_Time).dt.hour

# Extracting Minutes
train_data["Arrival_min"] = pd.to_datetime(train_data.Arrival_Time).dt.minute

# Now we can drop Arrival_Time as it is of no use
train_data.drop(["Arrival_Time"], axis = 1, inplace = True)


# In[80]:


train_data.head()


# In[81]:



# Assigning and converting Duration column into list
duration = list(train_data["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]           # Adds 0 hour

duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration


# In[82]:


train_data["Duration_hours"] = duration_hours
train_data["Duration_mins"] = duration_mins


# In[83]:


train_data.drop(["Duration"], axis = 1, inplace = True)


# In[84]:


train_data.head()


# In[85]:


train_data["Airline"].value_counts()


# In[86]:


Airline = train_data[["Airline"]]

Airline = pd.get_dummies(Airline, drop_first= True)

Airline.head()


# In[87]:


train_data["Source"].value_counts()


# In[98]:


sns.displot(train_data["Route"])
sns.displot(train_data["Duration_hours"])
plt.title("Price")


# In[100]:


import seaborn as sb
sb.set_context("notebook", font_scale=2.5)

from matplotlib import pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[105]:


sb.pairplot(train_data,size=3)


# In[106]:


Source = train_data[["Source"]]

Source = pd.get_dummies(Source, drop_first= True)

Source.head()


# In[107]:


train_data["Destination"].value_counts()


# In[108]:


Destination = train_data[["Destination"]]

Destination = pd.get_dummies(Destination, drop_first = True)

Destination.head()


# In[109]:


train_data["Route"]


# In[110]:


train_data.drop(["Route", "Additional_Info"], axis = 1, inplace = True)


# In[111]:


train_data["Total_Stops"].value_counts()


# In[112]:


train_data.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)


# In[113]:


train_data.head()


# In[114]:


data_train = pd.concat([train_data, Airline, Source, Destination], axis = 1)
data_train.head()


# In[115]:


data_train.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)
data_train.head()


# In[116]:


data_train.shape


# In[122]:


test_data = pd.read_excel("C:\\Users\\Lenovo\\Flight_Ticket_Participant_Datasets\\Data_Train.xlsx")
test_data.head()


# In[123]:


print("Test data Info")
print("-"*75)
print(test_data.info())

print()
print()

print("Null values :")
print("-"*75)
test_data.dropna(inplace = True)
print(test_data.isnull().sum())

# EDA

# Date_of_Journey
test_data["Journey_day"] = pd.to_datetime(test_data.Date_of_Journey, format="%d/%m/%Y").dt.day
test_data["Journey_month"] = pd.to_datetime(test_data["Date_of_Journey"], format = "%d/%m/%Y").dt.month
test_data.drop(["Date_of_Journey"], axis = 1, inplace = True)

# Dep_Time
test_data["Dep_hour"] = pd.to_datetime(test_data["Dep_Time"]).dt.hour
test_data["Dep_min"] = pd.to_datetime(test_data["Dep_Time"]).dt.minute
test_data.drop(["Dep_Time"], axis = 1, inplace = True)

# Arrival_Time
test_data["Arrival_hour"] = pd.to_datetime(test_data.Arrival_Time).dt.hour
test_data["Arrival_min"] = pd.to_datetime(test_data.Arrival_Time).dt.minute
test_data.drop(["Arrival_Time"], axis = 1, inplace = True)

# Duration
duration = list(test_data["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]           # Adds 0 hour

duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration

# Adding Duration column to test set
test_data["Duration_hours"] = duration_hours
test_data["Duration_mins"] = duration_mins
test_data.drop(["Duration"], axis = 1, inplace = True)


# Categorical data

print("Airline")
print("-"*75)
print(test_data["Airline"].value_counts())
Airline = pd.get_dummies(test_data["Airline"], drop_first= True)

print()

print("Source")
print("-"*75)
print(test_data["Source"].value_counts())
Source = pd.get_dummies(test_data["Source"], drop_first= True)

print()

print("Destination")
print("-"*75)
print(test_data["Destination"].value_counts())
Destination = pd.get_dummies(test_data["Destination"], drop_first = True)

# Additional_Info contains almost 80% no_info
# Route and Total_Stops are related to each other
test_data.drop(["Route", "Additional_Info"], axis = 1, inplace = True)

# Replacing Total_Stops
test_data.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)

# Concatenate dataframe --> test_data + Airline + Source + Destination
data_test = pd.concat([test_data, Airline, Source, Destination], axis = 1)

data_test.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)

print()
print()

print("Shape of test data : ", data_test.shape)


# In[124]:


data_test.head()


# In[125]:


data_train.shape


# In[126]:


data_train.columns


# In[128]:


X = data_train.loc[:,['Total_Stops', 'Journey_day', 'Journey_month', 'Dep_hour', 'Dep_min',
       'Arrival_hour', 'Arrival_min', 'Duration_hours', 'Duration_mins',
       'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo',
       'Airline_Jet Airways', 'Airline_Jet Airways Business',
       'Airline_Multiple carriers',
       'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
       'Airline_Vistara', 'Airline_Vistara Premium economy', 'Source_Chennai',
       'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai', 'Destination_Cochin',
       'Destination_Delhi', 'Destination_Hyderabad', 'Destination_Kolkata',
       'Destination_New Delhi']]
X.head()


# In[129]:


y = data_train.iloc[:, 1]
y.head()


# In[142]:


corr=data_train.corr()
plt.figure(figsize=(100,100))
sns.heatmap(data=corr,annot=True)


# In[150]:


from sklearn.ensemble import ExtraTreesRegressor
selection = ExtraTreesRegressor()
selection.fit(X,y)


# In[151]:


print(selection.feature_importances_)


# In[153]:


plt.figure(figsize = (12,8))
feat_importances = pd.Series(selection.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()

