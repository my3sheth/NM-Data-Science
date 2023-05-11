import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

car_dataset =pd.read_csv("C:\\Users\\admin\\Desktop\\Maitri\\Courses & Certifications\\Data Science\\Car Data.csv")

car_dataset.head()

# ### *1. EDA (Exploratory Data Analysis)*

car_dataset.shape

car_dataset.info()

categorical=car_dataset.select_dtypes(include=[object])
numerical=car_dataset.select_dtypes(include=[np.int32, np.int64, np.float64, np.float32])
print("Categorical features:",categorical.shape[1])
print("Numerical features:",numerical.shape[1])

car_dataset.describe()

car_dataset.isnull().sum()

car_dataset.head()

for i in numerical:
    sns.displot(x=car_dataset[i],kde=True)
plt.show()

car_dataset.head()

plt.figure(figsize=(100,100))
sns.barplot(y=car_dataset['Car_Name'],x=car_dataset['Selling_Price'])
plt.title('Car Names and their Selling Price')
plt.show()

car_dataset.head()

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

car_dataset.head()

plt.figure(figsize=(20,10))
sns.barplot(y=car_dataset['Selling_Price'],x=car_dataset['Year'])
plt.show()

car_dataset.head()

sns.scatterplot(y=car_dataset['Selling_Price'],x=car_dataset['Kms_Driven'])
plt.show()

car_dataset.head()

sns.barplot(y=car_dataset['Selling_Price'],x=car_dataset['Fuel_Type'])
plt.show()

car_dataset.head()

sns.barplot(y=car_dataset['Selling_Price'],x=car_dataset['Transmission'])
plt.show()

car_dataset.head()

sns.barplot(y=car_dataset['Selling_Price'],x=car_dataset['Owner'])
plt.show()

# ### *2. Preprocessing*

# #### (i) Encoding

from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()

car_dataset.head()

for i in ['Fuel_Type','Seller_Type','Transmission']:
    car_dataset[i]=encoder.fit_transform(car_dataset[i])
    print(car_dataset[i])

car_dataset.head()

car_dataset['Fuel_Type'].value_counts()

car_dataset.head()

Age = 2023-car_dataset['Year']
car_dataset.insert(1,"Age",Age)
car_dataset.drop('Year', axis = 1, inplace = True)

car_dataset.head()

car_dataset.columns

car_dataset.head()

# #### (ii) Outlier Detection and Handling

car_dataset.head()

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

car_dataset.shape

# ### *3. Train and Test*

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

car_dataset.head()

# ### *4. Feature Selection*

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
bestfeatures=SelectKBest(mutual_info_regression,k=5)
bestfeatures.fit(X_train.values,Y_train.values)
X_train_selected=X_train[X_train.columns[bestfeatures.get_support()]]

X_train_selected.columns

X_test_selected=X_test[X_test.columns[bestfeatures.get_support()]]

X_train_selected.shape

X_test_selected.shape

# ### *5. Model Selection*
# #### (i) Linear Regression

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
scaler1=MinMaxScaler()
scaler1.fit(X[['Present_Price']])
X['Present_Price']=scaler1.transform(X[['Present_Price']])
sns.displot(X.Present_Price,kde=True)
plt.show()

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train_selected,Y_train)
Y_pred=lr.predict(X_test_selected)

from sklearn import metrics
from sklearn.metrics import r2_score
r2=r2_score(Y_test,Y_pred)
print(r2)

mae=metrics.mean_absolute_error(Y_test,Y_pred)
mse=metrics.mean_squared_error(Y_test,Y_pred)
print("MAE is:",mae)
print("MSE is:",mse)

import math
from math import sqrt
rsme=sqrt(mse)
print("RSME is:",rsme)

# #### (ii) Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=200,max_depth=5,min_samples_leaf=100,n_jobs=4,random_state=22)
rf.fit(X_train_selected,Y_train)
Y_pred=rf.predict(X_test_selected)

from sklearn import metrics
from sklearn.metrics import r2_score
r2=r2_score(Y_test,Y_pred)
print(r2)

mae=metrics.mean_absolute_error(Y_test,Y_pred)
mse=metrics.mean_squared_error(Y_test,Y_pred)
print("MAE is:",mae)
print("MSE is:",mse)

rsme=sqrt(mse)
print("RSME is:",rsme)

from xgboost import XGBRegressor
xg=XGBRegressor()
xg.fit(X_train_selected,Y_train)
Y_pred=xg.predict(X_test_selected)

r2=r2_score(Y_test,Y_pred)
print(r2)

mae=metrics.mean_absolute_error(Y_test,Y_pred)
mse=metrics.mean_squared_error(Y_test,Y_pred)
print("MAE is:",mae)
print("MSE is:",mse)

rsme=sqrt(mae)
print("RSME is:",rsme)

# ### *6. Prediction*

for i in X_train_selected.columns:
    print(X_train_selected[i].value_counts())

print('Min.Preset_Price is:',X_train_selected.Present_Price.min())
print('Max.Preset_Price is:',X_train_selected.Present_Price.max())

# #### 1. Age: 5 - 17
# #### 2. Present_Price: varies
# #### 3. Kms_Driven: varies
# #### 4. Fuel_Type: 0 - CNG, 1 - Petrol, 2 - Diesel
# #### 5. Seller_Type: 0 - Dealer, 1 - Individual

features=X_train_selected.columns

X_train_selected.head()

inputs=[]
for f in features:
    f=float(input(f'Enter {f}:'))
    inputs.append(f)

i=np.array(inputs)
i=i.reshape(1,-1)
ans=xg.predict(i)
print('The selling price of this car is predicted as',ans[0])

