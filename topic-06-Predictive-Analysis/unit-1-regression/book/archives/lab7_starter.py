# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 09:20:52 2019

@author: Brenda_Mullally
"""

import pandas
import numpy
import statsmodels.api as sm
import statsmodels.formula.api as smf

#load dataset from the csv file in the dataframe called nesarc_data
nesarc_data = pandas.read_csv('nesarc_pds.csv',low_memory=False)

#create a dictionary of the variables you will use
var_in_use = [
    'TAB12MDX',
    'S3AQ3B1',
    'S3AQ3C1',
    'AGE'
    ]

#for each variable in the dictionary remove spaces, change to numeric and generate descriptive stats
for i in var_in_use:
    nesarc_data[i] = nesarc_data[i].replace(" ", numpy.NaN)
    nesarc_data[i] = pandas.to_numeric(nesarc_data[i])
    print('Descriptive stats for' +i)
    print(nesarc_data[i].describe())
    
    
#set PANDAS to show all columns in Data frame
pandas.set_option('display.max_columns', None)

#set PANDAS to show all rows in Data frame
pandas.set_option('display.max_rows', None)

# restrict to those observations that are between 18 and 25 and smoke now
nesarc_data = nesarc_data[(nesarc_data['AGE']>=18) & (nesarc_data['AGE']<=25) & (nesarc_data['CHECK321']=='1')]


# replacing missing data
#counts for S3AQ3B1
print('counts for S3AQ3B1 - usual frequency when smoked cigarettes')
print(nesarc_data['S3AQ3B1'].value_counts(sort=True))
nesarc_data['S3AQ3B1'].describe


#replace the value 9 in S3AQ3B1 with Nan to signify missing data
nesarc_data['S3AQ3B1']=nesarc_data['S3AQ3B1'].replace(9, numpy.NaN)
#check replacement worked
print((nesarc_data['S3AQ3B1']==9).sum())
print(nesarc_data['S3AQ3B1'].isnull().sum())

#counts for S3AQ3B1 after set to Nan
#print(nesarc_data.describe())
print('counts for S3AQ3B1 - usual frequency when smoked cigarettes')
print(nesarc_data['S3AQ3B1'].value_counts(sort=True))

#recoding values

#first create the dictionary to recode
recode1= {1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1}

#next use the map funciton to replace values using the recode dictionary
nesarc_data['USFREQ'] = nesarc_data['S3AQ3B1'].map(recode1)

#Now recode to a quantitative value based on an estimate of how many times per month each person smokes

recode2 = {1: 30, 2: 22, 3: 14, 4: 5, 5: 2.5, 6: 1}

nesarc_data['USFREQMO'] = nesarc_data['S3AQ3B1'].map(recode2)

#create new secondary variable to hold number of cigarettes per month
nesarc_data['NUMCIGMO_EST'] = nesarc_data['USFREQMO'] * nesarc_data['S3AQ3C1']
nesarc_data['NUMCIGMO_EST']=nesarc_data['NUMCIGMO_EST'].replace(" ", numpy.NaN)
nesarc_data['NUMCIGMO_EST'] = pandas.to_numeric(nesarc_data['NUMCIGMO_EST'])

