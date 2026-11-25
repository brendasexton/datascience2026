# -*- coding: utf-8 -*-
"""
Spyder Editor
Most recent update: 12/11/2024
This is a temporary script file.
"""

import pandas
import numpy

#load dataset from the csv file in the dataframe called nesarc_data
nesarc_data = pandas.read_csv('nesarc_pds.csv',low_memory=False)

#convert all columns names to uppercase
nesarc_data.columns = map(str.upper, nesarc_data.columns)
#nesarc_data.columns = map(str.lower, nesarc_data.columns)

#set PANDAS to show all columns in Data frame
pandas.set_option('display.max_columns', None)

#convert all columns names to uppercase
nesarc_data.columns = map(str.upper, nesarc_data.columns)


#bug fix for display formats to avoid run time errors
pandas.set_option('display.float_format',lambda x:'%f'%x)

#display the size of the dataset as the number of rows and columns
print (len(nesarc_data)) 
print (len(nesarc_data.columns))

#display the data types in the dataset
print(nesarc_data.dtypes)

#display general information about the dataset
nesarc_data.info()

#output the first 5 rows of data
nesarc_data.head(5)


#converting strings to numeric data for better output
nesarc_data['S3AQ3B1'] = pandas.to_numeric(nesarc_data['S3AQ3B1'], errors='coerce')
nesarc_data['S3AQ3C1'] = pandas.to_numeric(nesarc_data['S3AQ3C1'], errors='coerce')


"""This section begins by counting the values in a variable, frequency counts are the 
   first step in data analysis. The values counts are then also represented as percentages
   because many data consumers like to see both the raw counts and the percentages/proportions

"""

#Count the frequency of each value (0 or 1)  in the TAB12MDX variable
print('counts for TAB12MDX - nicotine dependence in the past 12 months, yes=1')
print(nesarc_data["TAB12MDX"].value_counts(sort=True))

#the same output can be achieved using the groupby function
print('groupby of TAB12MDX nicotine dependence in the past 12 months, yes=1')
print (nesarc_data.groupby('TAB12MDX').size())

#percentage output for TAB12MDX using value_counts() function
print('percentages for TAB12MDX - nicotine dependence in the past 12 months, yes=1')
print(nesarc_data["TAB12MDX"].value_counts(sort=True, normalize=True))


#percentage for TAB12MDX using the groupby() function
print('percentage of TAB12MDX nicotine dependence in the past 12 months, yes=1')
print(nesarc_data.groupby('TAB12MDX').size() * 100 / len(nesarc_data))


#Count of CHECK321
print('counts for CHECK321 - smoked in the past year, yes=1')
print(nesarc_data["CHECK321"].value_counts(sort=True))

#Percentages of CHECK321
print('percentages for CHECK321 - smoked in the past year, yes=1')
print(nesarc_data["CHECK321"].value_counts(sort=True, normalize=True))

#count of S3AQ3B1
print('counts for S3AQ3B1 - usual frequency when smoked cigarettes')
print(nesarc_data["S3AQ3B1"].value_counts(sort=True))

#percentages of S3AQ3B1
print('percentages for S3AQ3B1 - usual frequency when smoked cigarettes')
print(nesarc_data["S3AQ3B1"].value_counts(sort=True, normalize=True))

#counts of S3AQ3C1
print('counts for S3AQ3C1 - usual quanitity when smoked cigarettes')
print(nesarc_data["S3AQ3C1"].value_counts(sort=True))

#percentages of S3AQ3C1
print('percentages for S3AQ3C1 - usual quanitity when smoked cigarettes')
print(nesarc_data["S3AQ3C1"].value_counts(sort=True, normalize=True))


""" This section begins some cleaning/preparing of the data
"""

#Rename variable
nesarc_dict = {
'TAB12MDX': 'nicotine_dependence'
}

print('Data read... performing rename of columns...')
nesarc_data.rename(columns=nesarc_dict, inplace=True)

#Use new variable name
print('counts for nicotine dependence - nicotine dependence in the past 12 months, yes=1')
print(nesarc_data["nicotine_dependence"].value_counts(sort=True))