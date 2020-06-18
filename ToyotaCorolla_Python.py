# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 01:33:02 2020

@author: User
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns
import statsmodels.api as sm

df=pd.read_csv("E:\\Data Science\\Assignments\\Python code\\Multilinear_Regression\\ToyotaCorolla.csv",engine='python')
df
df.head()
df.columns
data_toyota=df[['Price','Age_08_04','KM','HP','cc','Doors','Gears','Quarterly_Tax','Weight']]
data_toyota=data_toyota.rename(columns={'Age_08_04':'AGE','cc':'CC','Doors':'DOORS','Quarterly_Tax':'TAX'})
data_toyota
#Here y=Price x=Age_08_04','KM','HP','cc','Doors','Gears','Quarterly_Tax','Weight'

data_toyota.corr()
type(data_toyota)

sns.pairplot(data_toyota)

model=smf.ols('Price~AGE+KM+HP+CC+DOORS+Gears+TAX+Weight',data=data_toyota).fit()
model.summary()

#CC and DOORS have observed p_value more than 0.05.So removing those and rebuilding model again

model2=smf.ols('Price~AGE+KM+HP+Gears+TAX+Weight',data=data_toyota).fit()
model2.summary()

sm.graphics.influence_plot(model)

#From Influence plot we can observe 80 is showing high influence so we can drop entire row

data_toyota_new=data_toyota.drop(data_toyota.index[[80]],axis=0)
data_toyota_new

#Now preparing new model for the data set data_toyota_new

model3=smf.ols('Price~AGE+KM+HP+CC+DOORS+Gears+TAX+Weight',data=data_toyota_new).fit()
model3.summary()

#After removing the influence factor .Still we are observing for Doors its p_value is more than 0.005.So,removing DOORS and constructing the final mode]

data_toyota_new.drop(['DOORS'],axis=0)
data_toyota_new
model_final=smf.ols('Price~AGE+KM+HP+CC+Gears+TAX+Weight',data=data_toyota_new).fit()
model_final.summary() #We can observe R-Sqaure value is 0.869



pred=model_final.predict(data_toyota_new)


plt.scatter(x=data_toyota_new['AGE+KM+HP+CC+Gears+TAX+Weight'],y=data_toyota_new['Price'],color='red');plt.plot(data_toyota_new['AGE+KM+HP+CC+Gears+TAX+Weight'],pred,color='black');plt.xlabel("AGE+KM+HP+CC+Gears+TAX+Weight");plt.ylabel("Price")


plt.scatter(data_toyota_new.Price,pred,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")
