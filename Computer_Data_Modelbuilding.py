# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 23:37:22 2020

@author: User
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns
import statsmodels.api as sm

df=pd.read_csv("E:\\Data Science\\Assignments\\Python code\\Multilinear_Regression\\Computer_Data.csv")

df=pd.get_dummies(df,columns=['cd','multi','premium'],drop_first=True)#To create dummie values
df
df = df.drop(columns="ID")
df.corr()
df.head()
df.columns
sns.pairplot(df)

model1=smf.ols("price~speed+hd+ram+screen+ads+trend+cd_yes+multi_yes+premium_yes",data=df).fit()#R-Square value = 0.776
model1.summary()
sm.graphics.influence_plot(model1)
df_new=df.drop(df.index[5960],axis=0)
df_new=df.drop(df.index[3783],axis=0)

model2=smf.ols("price~speed+hd+ram+screen+ads+trend+cd_yes+multi_yes+premium_yes",data=df_new).fit()#R-square value =0.775
model2.summary()

pred=model1.predict(df_new)
pred


model3=smf.ols("price~speed+hd+ram+screen+ads+trend",data=df).fit()#R-squre=0.712
model3.summary()
#Data vizualization
plt.scatter(df_new.price,pred,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")

#We can apply transformation method on variables to get better accuracy.

model4=smf.ols("price~np.log(speed+hd+ram+screen+ads+trend+cd_yes+multi_yes+premium_yes)",data=df).fit()
model4.summary()

#Exponential transformation
model5=smf.ols("np.log(price)~speed+hd+ram+screen+ads+trend+cd_yes+multi_yes+premium_yes",data=df).fit()
model5.summary()


