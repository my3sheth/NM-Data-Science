import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

car_dataset =pd.read_csv("C:\\Users\\admin\\Desktop\\Maitri\\Courses & Certifications\\Data Science\\Car Data.csv")
car_dataset.head()

# ## *1. EDA (Exploratory Data Analysis)*
car_dataset.shape
car_dataset.info()
car_dataset.describe()
car_dataset.isnull().sum()

#Visualizing significant data
ftype=car_dataset['Fuel_Type'].value_counts().sort_values(ascending=False)
print(ftype,'\n')
plt.pie(ftype,labels=['Petrol','Diesel','CNG'],autopct='%.2f',explode=[0.1,0.1,0.1])
plt.title('Fuel Type')
plt.show()

car_dataset.head()

seller=car_dataset['Seller_Type'].value_counts()
print(seller,'\n')
plt.bar(seller.index, seller.values,color=['green','orange'])
plt.title('Owner Type')
plt.show()

car_dataset.head()

transmission=car_dataset['Transmission'].value_counts()
print(transmission,'\n')
sns.barplot(x=transmission.index, y=transmission.values)
plt.title('Transmission')

# ### *2. Preprocessing*

# #### (i) Encoding

from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
car_dataset.head()

f_type=car_dataset[['Fuel_Type']]
car_dataset['Fuel_Type']=encoder.fit_transform(f_type)

car_dataset.head()

car_dataset['Fuel_Type'].value_counts()

car_dataset.head()

Age = 2023-car_dataset['Year']
car_dataset.insert(1,"Age",Age)
car_dataset.drop('Year', axis = 1, inplace = True)

car_dataset.head()

s_type=car_dataset[['Seller_Type']]
car_dataset['Seller_Type']=encoder.fit_transform(s_type)

car_dataset.head()

tran=car_dataset['Transmission']
car_dataset['Transmission']=encoder.fit_transform(tran)

car_dataset.head()

car_dataset.columns

# #### (ii) Outlier Detection and Handling

print(car_dataset.Age.value_counts(),'\n')
plt.boxplot(car_dataset.Age)
sns.kdeplot(car_dataset.Age)

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

#Handling Outliers for IQR Method
car_dataset.Age[car_dataset.Age<lower]=lower
car_dataset.Age[car_dataset.Age>upper]=upper

sns.kdeplot(car_dataset.Selling_Price)

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

#Handling Outliers for IQR Method
car_dataset.Selling_Price[car_dataset.Selling_Price<lower]=lower
car_dataset.Selling_Price[car_dataset.Selling_Price>upper]=upper

print(car_dataset.Present_Price.value_counts(),'\n')
plt.boxplot(car_dataset.Present_Price)

sns.kdeplot(car_dataset.Present_Price)

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

#Handling Outliers for IQR Method
car_dataset.Present_Price[car_dataset.Present_Price<lower]=lower
car_dataset.Present_Price[car_dataset.Present_Price>upper]=upper


print(car_dataset.Kms_Driven.value_counts(),'\n')
plt.boxplot(car_dataset.Kms_Driven)

sns.kdeplot(car_dataset.Kms_Driven)

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

#Handling Outliers for IQR Method
car_dataset.Kms_Driven[car_dataset.Kms_Driven<lower]=lower
car_dataset.Kms_Driven[car_dataset.Kms_Driven>upper]=upper

# ### *3. Train and Test*
car_dataset.shape

car_dataset.head()

X = car_dataset.drop(["Car_Name", "Selling_Price"], axis=1)
Y = car_dataset["Selling_Price"]

X.head()
Y.head()

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=2) 

X_train.shape

X_train.head()

Y_train.shape

Y_train.head()

X_test.shape

X_test.head()

Y_test.shape

Y_test.head()

