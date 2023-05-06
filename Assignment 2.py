#!/usr/bin/env python
# coding: utf-8

# In[1193]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


# In[1194]:


car_dataset =pd.read_csv("C:\\Users\\admin\\Desktop\\Maitri\\Courses & Certifications\\Data Science\\Car Data.csv")


# In[1195]:


car_dataset.head()


# ### *1. EDA (Exploratory Data Analysis)*
# 

# In[1196]:


car_dataset.shape


# In[1197]:


car_dataset.info()


# In[1198]:


car_dataset.describe()


# In[1199]:


car_dataset.isnull().sum()


# In[1200]:


ftype=car_dataset['Fuel_Type'].value_counts().sort_values(ascending=False)
print(ftype,'\n')
plt.pie(ftype,labels=['Petrol','Diesel','CNG'],autopct='%.2f',explode=[0.1,0.1,0.1])
plt.title('Fuel Type')
plt.show()


# In[1201]:


car_dataset.head()


# In[1202]:


seller=car_dataset['Seller_Type'].value_counts()
print(seller,'\n')
plt.bar(seller.index, seller.values,color=['green','orange'])
plt.title('Owner Type')
plt.show()


# In[1203]:


car_dataset.head()


# In[1204]:


transmission=car_dataset['Transmission'].value_counts()
print(transmission,'\n')
sns.barplot(x=transmission.index, y=transmission.values)
plt.title('Transmission')


# ## *2. Preprocessing*

# #### (i) Encoding

# In[1205]:


from sklearn.preprocessing import LabelEncoder


# In[1206]:


encoder=LabelEncoder()


# In[1207]:


car_dataset.head()


# In[1208]:


f_type=car_dataset[['Fuel_Type']]
car_dataset['Fuel_Type']=encoder.fit_transform(f_type)


# In[1209]:


car_dataset.head()


# In[1210]:


car_dataset['Fuel_Type'].value_counts()


# In[1211]:


car_dataset.head()


# In[1212]:


Age = 2023-car_dataset['Year']
car_dataset.insert(1,"Age",Age)
car_dataset.drop('Year', axis = 1, inplace = True)


# In[1213]:


car_dataset.head()


# In[1214]:


s_type=car_dataset[['Seller_Type']]
car_dataset['Seller_Type']=encoder.fit_transform(s_type)


# In[1215]:


car_dataset.head()


# In[1216]:


tran=car_dataset['Transmission']
car_dataset['Transmission']=encoder.fit_transform(tran)


# In[1217]:


car_dataset.head()


# In[1218]:


car_dataset.columns


# #### (ii) Outlier Detection and Handling

# In[1234]:


#Age
print(car_dataset.Age.value_counts(),'\n')
plt.boxplot(car_dataset.Age)


# In[1231]:


sns.kdeplot(car_dataset.Age)


# In[1232]:


#Detecing Outliers using IQR Method (for skewed distribution)
age=car_dataset.Age
q1=age.quantile(0.25)
q3=age.quantile(0.75)
iqr=q3-q1
lower=q1-1.5*iqr
upper=q3+1.5*iqr

age[(age<lower)|(age>upper)]

print(lower)
print(upper)


# In[1233]:


#Handling Outliers for IQR Method
car_dataset.Age[car_dataset.Age<lower]=lower
car_dataset.Age[car_dataset.Age>upper]=upper


# In[1177]:


sns.kdeplot(car_dataset.Selling_Price)


# In[1178]:


#Detecing Outliers using IQR Method (for skewed distribution)
sp=car_dataset.Selling_Price
q1=sp.quantile(0.25)
q3=sp.quantile(0.75)
iqr=q3-q1
lower=q1-1.5*iqr
upper=q3+1.5*iqr

sp[(sp<lower)|(sp>upper)]

print(lower)
print(upper)


# In[1179]:


#Handling Outliers for IQR Method
car_dataset.Selling_Price[car_dataset.Selling_Price<lower]=lower
car_dataset.Selling_Price[car_dataset.Selling_Price>upper]=upper


# In[1235]:


print(car_dataset.Present_Price.value_counts(),'\n')
plt.boxplot(car_dataset.Present_Price)


# In[1236]:


sns.kdeplot(car_dataset.Present_Price)


# In[1237]:


#Detecing Outliers using IQR Method (for skewed distribution)
pp=car_dataset.Present_Price
q1=pp.quantile(0.25)
q3=pp.quantile(0.75)
iqr=q3-q1
lower=q1-1.5*iqr
upper=q3+1.5*iqr

pp[(pp<lower)|(pp>upper)]

print(lower)
print(upper)


# In[1238]:


#Handling Outliers for IQR Method
car_dataset.Present_Price[car_dataset.Present_Price<lower]=lower
car_dataset.Present_Price[car_dataset.Present_Price>upper]=upper


# In[1239]:


print(car_dataset.Kms_Driven.value_counts(),'\n')
plt.boxplot(car_dataset.Kms_Driven)


# In[1240]:


sns.kdeplot(car_dataset.Kms_Driven)


# In[1241]:


#Detecing Outliers using IQR Method (for skewed distribution)
km=car_dataset.Kms_Driven
q1=km.quantile(0.25)
q3=km.quantile(0.75)
iqr=q3-q1
lower=q1-1.5*iqr
upper=q3+1.5*iqr

km[(km<lower)|(km>upper)]

print(lower)
print(upper)


# In[1266]:


#Handling Outliers for IQR Method
car_dataset.Kms_Driven[car_dataset.Kms_Driven<lower]=lower
car_dataset.Kms_Driven[car_dataset.Kms_Driven>upper]=upper


# In[1267]:


car_dataset.shape


# In[1268]:


car_dataset.head()


# In[1244]:


X = car_dataset.drop(["Car_Name", "Selling_Price"], axis=1)
Y = car_dataset["Selling_Price"]


# In[1245]:


X.head()


# In[1246]:


Y.head()


# In[1251]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=2) 


# In[1256]:


X_train.shape


# In[1253]:


X_train.head()


# In[1257]:


Y_train.shape


# In[1259]:


Y_train.head()


# In[1261]:


X_test.shape


# In[1262]:


X_test.head()


# In[1263]:


Y_test.shape


# In[1264]:


Y_test.head()

