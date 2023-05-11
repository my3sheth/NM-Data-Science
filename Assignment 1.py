import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

titanic=sns.load_dataset("titanic")
titanic.head()

#Question 1
print("1. What is the overall survival rate of passengers on the Titanic?\n")
survival=pd.Series(titanic['alive'].value_counts())
print(survival,"\n")
#Visualizing the filtered data
plt.pie(survival,labels=['dead','alive'],autopct="%.2f")
plt.legend()
plt.title('Survival % of Passengers')
plt.show()

titanic.head()

#Question 2
print("2. What is the gender distribution among the passengers on the Titanic?\n")
gender_dist=pd.Series(titanic['sex'].value_counts())
print(gender_dist,"\n")
#Visualizing the filtered data
plt.title("Gender Distribution")
plt.xlabel("sex")
plt.ylabel("no. of passengers")
sns.barplot(x=gender_dist.index, y=gender_dist.values, data=titanic)

titanic.head()

#Question 3
print("3. Did the survival rate differ by gender? If so, by how much?\n")
gender_surv=titanic.groupby('sex')['alive'].value_counts()
print(gender_surv,'\n')
#Visualizing the filtered data
plt.title("Gender Distribution Based on Survival")
sns.barplot(x='sex', y='survived', data=titanic)

titanic.head()

#Question 4
print("4. What was the age distribution among the passengers on the Titanic?\n")
age_dist=pd.Series(titanic['age'].value_counts().sort_values(ascending=False))
print(age_dist,'\n')
#Visualizing the filtered data
plt.hist(titanic['age'],bins=10,linewidth=1.5,linestyle='-',edgecolor='white',color='green')
plt.title('Age Dsitribution')
plt.xlabel('age')
plt.ylabel('no. of Passengers')
plt.show()

titanic.head()

#Question 5
print("5. Did the survival rate differ by the journey class? If so, by how much?\n")
pclass_surv=titanic.groupby('pclass')['alive'].value_counts()
print(pclass_surv,'\n')
#Visualizing the filtered data
plt.title("Survival Rate Based on Journey Class")
sns.barplot(x='pclass', y='survived', data=titanic)


