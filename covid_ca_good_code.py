code to use in labs

2022 ca on covid data

Liangyu Chen

# (1) Load data.
# (2) low_memory=False to increase efficiency, preventing Pandas from determining data type for each variables.
covid_data = pandas.read_csv('owid-covid-data.csv', low_memory=False)

# avoid runtime error
pandas.set_option('display.float_format',lambda x:'%f'%x)

# set PANDAS to show all columns in Data frame
pandas.set_option('display.max_columns', None)

# set PANDAS to show all rows in Data frame
pandas.set_option('display.max_rows', None)

# convert all columns names to lowercase
covid_data.columns = map(str.lower, covid_data.columns)

def removeEmpty(subset,columnName):
    subset[columnName] = subset[columnName].replace(' ', numpy.NaN)
    subset[columnName] = subset[columnName].replace('', numpy.NaN)
    # fillna() - replace nulls to a specified value
    # https://thepythonyouneed.com/how-to-replace-null-values-using-pandas/
    subset[columnName] = subset[columnName].fillna(numpy.NaN)

def toNumeric(subset,columnName):
    subset[columnName] = pandas.to_numeric(subset[columnName],errors='coerce')

def printDistribution(subset,columnName):
    print(subset[columnName].describe())
    print(subset[columnName].value_counts(sort=True))
    # print(subset[columnName].value_counts(sort=True, normalize=True))
    print()

def printSize(subset):
    print(len(subset),'rows')
    print(len(subset.columns),'columns')
    print()




print("Number of rows and columns of covid_data.")
printSize(covid_data)

data = covid_data[(covid_data['iso_code']=='USA')]
data = data.copy()
selected_columns = ['date','total_cases','new_cases','total_deaths','new_deaths','total_vaccinations','new_vaccinations','people_vaccinated','people_fully_vaccinated','population']
data = pandas.DataFrame(data, columns = selected_columns)

print("Number of rows and columns of selected data.")
printSize(data)




# data manage
for i in selected_columns[1:]:
    removeEmpty(data,i)
    toNumeric(data,i)
    print('Descriptive stats for ' + i)
    print(data[i].describe())
    print()



-----------------------------------------------------------------------------




Teek Leng has best CA on covid data, covered all aspects.

this gets average positive rate for each location,
subset_eu2=subset_eu2[['location', 'population_density', 'human_development_index', 'positive_rate']].dropna()

subset_eu2 = subset_eu2.groupby(['location', 'population_density', 'human_development_index'])['positive_rate'].mean()
subset_eu2 = subset_eu2.reset_index()
subset_eu2.rename(columns = {'positive_rate':'average_positive_rate'}, inplace=True)
print(subset_eu2)



----------------------------------------------------------------------------

#divide the time object into three columns of year, month, and day data
covid_data['date'] = pandas.to_datetime(covid_data['date'])
covid_data["year"] = covid_data['date'].apply(lambda x:x.year)
covid_data["month"] = covid_data['date'].apply(lambda x:x.month)
covid_data["day"] = covid_data['date'].apply(lambda x:x.day)

#Combine countries subsets
subset_USA = subset1[(subset1['location']=='United States')]
subset_IRE = subset1[(subset1['location']=='Ireland')]
subset_UKD = subset1[(subset1['location']=='United Kingdom')]
subset_CAN = subset1[(subset1['location']=='Canada')]

subsets = pandas.DataFrame()
subsets.reset_index(drop=True)
subsets = subsets.append(subset_USA)
subsets = subsets.append(subset_IRE)
subsets = subsets.append(subset_UKD)
subsets = subsets.append(subset_CAN)

#Find the data related to Mexico in the vaccine data, and convert the date feature to datetime type
vaccinations_data = data[data['iso_code']=='MEX']
vaccinations_data['date'] = vaccinations_data['date'].apply(lambda x:pd.to_datetime(x))
#The overall data is filtered for data related to Mexico, and the date is converted to datetime type to filter the data of the last six months
mex_data = whole_data[whole_data['iso_code']=='MEX']
mex_data['date'] = mex_data['date'].apply(lambda x:pd.to_datetime(x))
aggregate_data = mex_data[(mex_data['date']>='2020-12-24') & (mex_data['date'] <= '2021-05-09')]
summary_data = pd.merge(aggregate_data,vaccinations_data,on='date')


print('--------------- new varibale --------------')
def count_days_from_start(tempDf0):
    tempDF = tempDf0.copy()
    tempDF.reset_index(drop=True,inplace=True)
    tempDF['date'] = tempDF['date'].apply(lambda x:datetime.strptime(x,r'%Y-%m-%d'))
    tempDF['date2'] = tempDF.loc[0,'date']
    tempDF['days'] = tempDF.apply(lambda x:(x['date']-x['date2']).days,axis=1)
    return tempDF

France_Df = df[df['location']=='France']
France_Df = count_days_from_start(France_Df)



#Aggregated mortality and bed rates for each country
df2 = df.groupby('location').agg({'total_cases_per_million':'max','hospital_beds_per_thousand':'mean','total_deaths_per_million':'max','extreme_poverty':'mean','aged_65_older':'mean'})
print(df2.total_cases_per_million.describe())




#-------Set all numbers to two decimal places----------
pandas.options.display.float_format = '{:,.2f}'.format

Tan Shi
# Seeking top10 countries
location_df = data.groupby(by = 'location')['total_cases'].sum().reset_index()
location_df.sort_values(by='total_cases',ascending=False,inplace=True)
location_df.head(10)

#Remove data from continents
continents = ['World','Europe','North America','Asia','European Union','South America','Africa','South Africa']
country_df = location_df[~location_df['location'].isin(continents)].reset_index(drop=True)
top10 = country_df.head(10)

# Get other countries total case data
top10_countries = list(top10['location'])
not_top10_country = country_df[~country_df['location'].isin(top10_countries)]
not_top10_country_total = not_top10_country['total_cases'].sum()
not_top10_country_series = pd.Series({'location':'not_top10_country', 'total_cases':not_top10_country_total})
top10_other_df = top10.append(not_top10_country_series,ignore_index=True)

# The proportion of top10 countries' infections relative to the global
plt.figure(figsize=(10, 8))
plt.pie(top10_other_df['total_cases'],labels=top10_other_df['location'],autopct='%1.1f%%',shadow=False,startangle=150)
plt.title("top10 country rate")

-------------------------------------------------------------------------------------
li Zhixiang
#write for loop to conduct frequency dist for a list of variables
var_pick_final=[
'weekly_hosp_admissions',
'weekly_hosp_admissions_per_million',
'icu_patients_per_million',
'hosp_patients',
'hosp_patients_per_million',
'weekly_icu_admissions',
'weekly_icu_admissions_per_million',
'new_vaccinations',
'new_vaccinations_smoothed',
'total_vaccinations_per_hundred',
'people_vaccinated_per_hundred',
'people_fully_vaccinated_per_hundred',
 'total_vaccinations',
'people_fully_vaccinated' ]

#2.frequency distributions
import matplotlib.pyplot as plt
for i in var_pick_final:
    dtf[i].hist()
    plt.title(i)
    plt.show()



    -----------------------------
    #create a new dataframe to store description data -- country
description = pandas.DataFrame()
newlist = []
for i in subset2['location'].drop_duplicates():
    newlist.append(i)
description.insert(0,'location',newlist)
print(description['location'])

--------------------------------------------------
# 1-2.Screening data of China and India from January 1, 2020 to December 31, 2020
def get_data():
    df_source = pd.read_csv("owid-covid-data.csv")
    df_input = df_source[['continent','location','date','total_cases','new_cases','total_deaths','total_tests','population','human_development_index']]
    df_input = df_input.where(df_input['date']>='2020-01-01')
    df_input = df_input.where(df_input['date']<='2020-12-31').dropna(axis=0,how="all") # 60139行数据
    df_china = df_input[df_input['location']=="China"] #345
    df_india = df_input[df_input['location']=="India"] #337
    #df_china.to_csv('data/china.csv',header=True, index=False)
    #df_india.to_csv('data/india.csv',header=True, index=False)

    #3. Calculate monthly incidence rate and monthly mortality rate
    df_china_india = pd.concat([df_china,df_india])

    df_china_india['date'] = df_china_india['date'].apply(lambda x:x[:7])
    df_china_india_month = df_china_india.groupby(["location","date"]).agg({'total_cases':'sum','new_cases':'sum','total_deaths':'sum','population':'max'})

    df_china_india_month['month_incidence_rate'] = df_china_india_month[["total_cases","population"]].apply(lambda x:x["total_cases"]/x["population"],axis=1)
    df_china_india_month['month_death_rate'] = df_china_india_month[["total_deaths","population"]].apply(lambda x:x["total_deaths"]/x["population"],axis=1)

    # df_china_india_month.to_csv('data/china_india_month_data.csv',header=True, index=False)
    return df_china_india_month
    -----------------------------------------------
    |Zhengjie Ye

    choosed_variables=['reproduction_rate','total_tests_per_thousand','new_tests_per_thousand','tests_per_case','total_vaccinations_per_hundred','people_vaccinated_per_hundred',
                 'people_fully_vaccinated_per_hundred']
subset_q2=covid[choosed_variables].copy()

# data management
# print out the type of all variables
print("Type of each variable:")
print(subset_q2.dtypes)


#Count and percentage for each variable using value_counts function
for name in choosed_variables:
    print('counts for '+ name +' - '+ name)
    c= subset_q2[name].value_counts(sort=True, dropna=False)
    print (c)

    print('percentages for '+ name +' - '+name)
    p= subset_q2[name].value_counts(sort=True, normalize=True)
    print (p)

for name in choosed_variables:
    print("\nThe count of null of "+ name)
    print(subset_q2[name].isnull().sum())
for name in choosed_variables:
    print("\nThe count of '' of "+ name)
    print(subset_q2[subset_q2[name]==''].sum())
for name in choosed_variables:
    print("\nThe count of ' ' of "+ name)
    print(subset_q2[subset_q2[name]==' '].sum())


    def R_classificate (row):
    # The basic reproduction number, or basic reproductive number (sometimes called basic reproduction ratio or basic reproductive rate), denoted R_{0} (pronounced R nought or R zero),
    # when R0>1 the infection will be able to start spreading in a population, but not if R0<1. Generally, the larger the value of R0, the harder it is to control the epidemic.
    # https://en.wikipedia.org/wiki/Basic_reproduction_number
    # 1 indicate covid is still spreading
    if row['reproduction_rate']>=1:
        return 1
    # 0 indicate covid will gradually disappear
    if row['reproduction_rate'] <1:
        return 0

# create new secondary variable to group the observation to different R classification
subset_q2['R_classification'] = subset_q2.apply(lambda row: R_classificate (row), axis=1)
print('counts for R_classification')
c= subset_q2['R_classification'].value_counts(sort=True, dropna=False)
print (c)

    -------------------------------------------------------------------------------

    Zhouhao Rui

    # change the date time, and get the data in 2021
covid_data['date'] = pandas.to_datetime(covid_data['date'])
covid_data = covid_data.set_index('date')
covid_data['date'] = covid_data.index
covid_data = covid_data['2021-1': '2021-5']
