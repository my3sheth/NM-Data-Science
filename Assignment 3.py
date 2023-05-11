#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


car_dataset =pd.read_csv("C:\\Users\\admin\\Desktop\\Maitri\\Courses & Certifications\\Data Science\\Car Data.csv")


# In[3]:


car_dataset.head()


# ### *1. EDA (Exploratory Data Analysis)*
# 

# In[4]:


car_dataset.shape


# In[5]:


car_dataset.info()


# In[6]:


#del car_dataset['Owner']


# In[7]:


categorical=car_dataset.select_dtypes(include=[object])
numerical=car_dataset.select_dtypes(include=[np.int32, np.int64, np.float64, np.float32])
print("Categorical features:",categorical.shape[1])
print("Numerical features:",numerical.shape[1])


# In[8]:


car_dataset.describe()


# In[9]:


car_dataset.isnull().sum()


# In[10]:


car_dataset.head()


# In[11]:


for i in numerical:
    sns.displot(x=car_dataset[i],kde=True)
plt.show()


# In[12]:


car_dataset.head()


# In[13]:


plt.figure(figsize=(100,100))
sns.barplot(y=car_dataset['Car_Name'],x=car_dataset['Selling_Price'])
plt.title('Car Names and their Selling Price')
plt.show()


# In[14]:


car_dataset.head()


# In[15]:


ftype=car_dataset['Fuel_Type'].value_counts().sort_values(ascending=False)
print(ftype,'\n')
plt.pie(ftype,labels=['Petrol','Diesel','CNG'],autopct='%.2f',explode=[0.1,0.1,0.1])
plt.title('Fuel Type')
plt.show()


# In[16]:


car_dataset.head()


# In[17]:


seller=car_dataset['Seller_Type'].value_counts()
print(seller,'\n')
plt.bar(seller.index, seller.values,color=['green','orange'])
plt.title('Owner Type')
plt.show()


# In[18]:


car_dataset.head()


# In[19]:


transmission=car_dataset['Transmission'].value_counts()
print(transmission,'\n')
sns.barplot(x=transmission.index, y=transmission.values)
plt.title('Transmission')


# In[20]:


car_dataset.head()


# In[21]:


plt.figure(figsize=(20,10))
sns.barplot(y=car_dataset['Selling_Price'],x=car_dataset['Year'])
plt.show()


# In[22]:


car_dataset.head()


# In[23]:


sns.scatterplot(y=car_dataset['Selling_Price'],x=car_dataset['Kms_Driven'])
plt.show()


# In[24]:


car_dataset.head()


# In[25]:


sns.barplot(y=car_dataset['Selling_Price'],x=car_dataset['Fuel_Type'])
plt.show()


# In[26]:


car_dataset.head()


# In[27]:


sns.barplot(y=car_dataset['Selling_Price'],x=car_dataset['Transmission'])
plt.show()


# In[28]:


car_dataset.head()


# In[29]:


sns.barplot(y=car_dataset['Selling_Price'],x=car_dataset['Owner'])
plt.show()


# ## *2. Preprocessing*

# #### (i) Encoding

# In[30]:


from sklearn.preprocessing import LabelEncoder


# In[31]:


encoder=LabelEncoder()


# In[32]:


car_dataset.head()


# In[33]:


for i in ['Fuel_Type','Seller_Type','Transmission']:
    car_dataset[i]=encoder.fit_transform(car_dataset[i])
    print(car_dataset[i])


# In[34]:


car_dataset.head()


# In[35]:


car_dataset['Fuel_Type'].value_counts()


# In[36]:


car_dataset.head()


# In[37]:


Age = 2023-car_dataset['Year']
car_dataset.insert(1,"Age",Age)
car_dataset.drop('Year', axis = 1, inplace = True)


# In[38]:


car_dataset.head()


# In[39]:


car_dataset.columns


# In[40]:


car_dataset.head()


# #### (ii) Outlier Detection and Handling

# In[41]:


car_dataset.head()


# In[42]:


#Age
print(car_dataset.Age.value_counts(),'\n')
plt.boxplot(car_dataset.Age)


# In[43]:


sns.kdeplot(car_dataset.Age)


# In[44]:


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


# In[45]:


#Handling Outliers for IQR Method
car_dataset.Age[car_dataset.Age<lower]=lower
car_dataset.Age[car_dataset.Age>upper]=upper


# In[46]:


sns.kdeplot(car_dataset.Selling_Price)


# In[47]:


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


# In[48]:


#Handling Outliers for IQR Method
car_dataset.Selling_Price[car_dataset.Selling_Price<lower]=lower
car_dataset.Selling_Price[car_dataset.Selling_Price>upper]=upper


# In[49]:


print(car_dataset.Present_Price.value_counts(),'\n')
plt.boxplot(car_dataset.Present_Price)


# In[50]:


sns.kdeplot(car_dataset.Present_Price)


# In[51]:


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


# In[52]:


#Handling Outliers for IQR Method
car_dataset.Present_Price[car_dataset.Present_Price<lower]=lower
car_dataset.Present_Price[car_dataset.Present_Price>upper]=upper


# In[53]:


print(car_dataset.Kms_Driven.value_counts(),'\n')
plt.boxplot(car_dataset.Kms_Driven)


# In[54]:


sns.kdeplot(car_dataset.Kms_Driven)


# In[55]:


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


# In[56]:


#Handling Outliers for IQR Method
car_dataset.Kms_Driven[car_dataset.Kms_Driven<lower]=lower
car_dataset.Kms_Driven[car_dataset.Kms_Driven>upper]=upper


# In[57]:


car_dataset.shape


# ### *3. Train and Test*

# In[58]:


car_dataset.head()


# In[59]:


X = car_dataset.drop(["Car_Name", "Selling_Price"], axis=1)
Y = car_dataset["Selling_Price"]


# In[60]:


X.head()


# In[61]:


Y.head()


# In[62]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=2) 


# In[63]:


X_train.shape


# In[64]:


X_train.head()


# In[65]:


Y_train.shape


# In[66]:


Y_train.head()


# In[67]:


X_test.shape


# In[68]:


X_test.head()


# In[69]:


Y_test.shape


# In[70]:


Y_test.head()


# In[71]:


car_dataset.head()


# ### *4. Feature Selection*

# In[72]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
bestfeatures=SelectKBest(mutual_info_regression,k=5)
bestfeatures.fit(X_train.values,Y_train.values)
X_train_selected=X_train[X_train.columns[bestfeatures.get_support()]]


# In[73]:


X_train_selected.columns


# In[74]:


X_test_selected=X_test[X_test.columns[bestfeatures.get_support()]]


# In[75]:


X_train_selected.shape


# In[76]:


X_test_selected.shape


# ### *5. Model Selection*
# #### (i) Linear Regression

# In[77]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
scaler1=MinMaxScaler()
scaler1.fit(X[['Present_Price']])
X['Present_Price']=scaler1.transform(X[['Present_Price']])
sns.displot(X.Present_Price,kde=True)
plt.show()


# In[78]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train_selected,Y_train)
Y_pred=lr.predict(X_test_selected)


# In[79]:


from sklearn import metrics
from sklearn.metrics import r2_score
r2=r2_score(Y_test,Y_pred)
print(r2)


# In[80]:


mae=metrics.mean_absolute_error(Y_test,Y_pred)
mse=metrics.mean_squared_error(Y_test,Y_pred)
print("MAE is:",mae)
print("MSE is:",mse)


# In[81]:


import math
from math import sqrt
rsme=sqrt(mse)
print("RSME is:",rsme)


# #### (ii) Random Forest Regressor

# In[82]:


from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=200,max_depth=5,min_samples_leaf=100,n_jobs=4,random_state=22)
rf.fit(X_train_selected,Y_train)
Y_pred=rf.predict(X_test_selected)


# In[83]:


from sklearn import metrics
from sklearn.metrics import r2_score
r2=r2_score(Y_test,Y_pred)
print(r2)


# In[84]:


mae=metrics.mean_absolute_error(Y_test,Y_pred)
mse=metrics.mean_squared_error(Y_test,Y_pred)
print("MAE is:",mae)
print("MSE is:",mse)


# In[85]:


rsme=sqrt(mse)
print("RSME is:",rsme)


# In[86]:


from xgboost import XGBRegressor
xg=XGBRegressor()
xg.fit(X_train_selected,Y_train)
Y_pred=xg.predict(X_test_selected)


# In[87]:


r2=r2_score(Y_test,Y_pred)
print(r2)


# In[88]:


mae=metrics.mean_absolute_error(Y_test,Y_pred)
mse=metrics.mean_squared_error(Y_test,Y_pred)
print("MAE is:",mae)
print("MSE is:",mse)


# In[89]:


rsme=sqrt(mae)
print("RSME is:",rsme)


# ### *6. Prediction*

# In[90]:


for i in X_train_selected.columns:
    print(X_train_selected[i].value_counts())


# In[91]:


print('Min.Preset_Price is:',X_train_selected.Present_Price.min())
print('Max.Preset_Price is:',X_train_selected.Present_Price.max())


# #### 1. Age: 5 - 17
# #### 2. Present_Price: varies
# #### 3. Kms_Driven: varies
# #### 4. Fuel_Type: 0 - CNG, 1 - Petrol, 2 - Diesel
# #### 5. Seller_Type: 0 - Dealer, 1 - Individual

# In[92]:


features=X_train_selected.columns


# In[93]:


X_train_selected.head()


# In[104]:


inputs=[]
for f in features:
    f=float(input(f'Enter {f}:'))
    inputs.append(f)


# In[105]:


i=np.array(inputs)
i=i.reshape(1,-1)
ans=xg.predict(i)
print('The selling price of this car is predicted as',ans[0])

