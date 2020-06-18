# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 16:25:59 2020

@author: Rohith
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm

df=pd.read_csv("E:\Data Science\Assignments\Python code\Multilinear_Regression\\50_Startups.csv")
df=pd.get_dummies(df,columns=['State'],drop_first=True)
df=df.rename(columns={'R&D Spend':'RD','Marketing Spend':'Marketing','State_New York':'Newyork','State_Florida':'Florida'})
df
df.head()
df.columns
type(df)
sns.pairplot(df)

#y=profit ; 

model=smf.ols("Profit~RD+Marketing+Administration+Florida+Newyork",data=df).fit()
model.summary()
model.params

sm.graphics.influence_plot(model)

df_new=df.drop(df.index[[46,48,49]],axis=0)

model2=smf.ols("Profit~RD+Marketing+Administration+Florida+Newyork",data=df_new).fit()
model2.summary()

#Applying transformation method for getting better predictions
#
model3=smf.ols("Profit~np.log(RD+Marketing+Administration+Florida+Newyork)",data=df_new).fit()
model3.summary()
#Exponential transformation
model4=smf.ols("np.log(Profit)~RD+Marketing+Administration+Florida+Newyork",data=df_new).fit()
model4.summary()

#Deleting State variable

model5=smf.ols("Profit~RD+Marketing+Administration",data=df_new).fit()
model5.summary()

#Again applying transformations
model6=smf.ols("np.log(Profit)~RD+Marketing+Administration",data=df_new).fit()
model6.summary()

model7=smf.ols("Profit~np.log(RD+Marketing+Administration)",data=df_new).fit()
model7.summary()

#Deleting Administration variable

model8=smf.ols("Profit~RD+Marketing",data=df_new).fit()
model8.summary()

#Again applying transformations
model9=smf.ols("np.log(Profit)~RD+Marketing",data=df_new).fit()
model9.summary()







