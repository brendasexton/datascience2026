# -*- coding: utf-8 -*-
"""
updated 15/03/2022

@author: B.Mullally
"""
import numpy as numpyp
import pandas as pandas
import statsmodels.api
import statsmodels.formula.api as smf
import seaborn 
import matplotlib.pyplot as plt

# bug fix for display formats to avoid run time errors
pandas.set_option('display.float_format', lambda x:'%.2f'%x)

#call in data set
data = pandas.read_csv('gapminder.csv')

vars_in_use= [
    'internetuserate',
    'urbanrate',
    'relectricperperson'
    ]
for i in vars_in_use:
    # convert variables to numeric format using convert_objects function
    data[i] = pandas.to_numeric(data[i], errors='coerce')
    print('Descriptive stats for ' +i)
    print(data[i].describe())
    

#Scatter plot to see relationship between urbanrate and internetuserate
scat1 = seaborn.regplot(x="urbanrate", y="internetuserate", scatter=True, data=data)
plt.xlabel('Urbanization Rate')
plt.ylabel('Internet Use Rate')
plt.title ('Scatterplot for the Association Between Urban Rate and Internet Use Rate')
print(scat1)

#regression model for urbanrate and internetuserate
print ("OLS regression model for the association between urban rate and internet use rate")
reg1 = smf.ols('internetuserate ~ urbanrate', data=data).fit()
print (reg1.summary())

# predict value of internetuserate where urbanrate is 80
# y = mx + c
# y = .72(80) + (-4.90)
# y = 57.6 + (-4.90)
# y = 52.7
# A rate of 52.7 internet use is predicted.

#Scatter plot to see relationship between urbanrate and relectricperperson
scat1 = seaborn.regplot(x="urbanrate", y="relectricperperson", scatter=True, data=data)
plt.xlabel('Urbanization Rate')
plt.ylabel('Electric use Rate')
plt.title ('Scatterplot for the Association Between Urban Rate and Electric Use Rate')
print(scat1)

#regression model for between urbanrate and relectricperperson
print ("OLS regression model for the association between urban rate and Electric use rate")
reg1 = smf.ols('relectricperperson ~ urbanrate', data=data).fit()
print (reg1.summary())

# predict value of relectricuserate where urbanrate is 80
# y = mx + c
# y = 37.78(80) + (-1151.66)
# y = 3022.4 + (-1151.66)
# y = 1870.74
# A rate of 1870.74 electric use is predicted.


