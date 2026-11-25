# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 10:54:56 2021

@author: BMULLALLY
"""

## for data
import pandas as pd
import numpy as np
## for plotting
import matplotlib.pyplot as plt
import seaborn as sns
## for statistical tests
import scipy
import statsmodels.formula.api as smf
import statsmodels.api as sm
## for machine learning
from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition
## for explainer
from lime import lime_tabular
import math

#read in the data
dtf = pd.read_csv('data_titanic.csv')
dtf.head()


#convert all columns names to uppercase
dtf.columns = map(str.upper, dtf.columns)


#set PANDAS to show all columns in Data frame
pd.set_option('display.max_columns', None)

#bug fix for display formats to avoid run time errors
pd.set_option('display.float_format',lambda x:'%f'%x)

print(dtf.describe())
print(dtf.dtypes)
print (len(dtf)) 
print (len(dtf.columns))

#rename the classification variable to Y
dtf = dtf.rename(columns={"SURVIVED":"Y"})

bc1 = sns.countplot(x='Y',data=dtf)
plt.xlabel('Number of Passengers')
plt.title('Passenger Survvial rates')

#count of Y variable
print('count of Y Variable')
c1 = dtf.groupby('Y').size()
print(c1)

#percentages for Y variable
print('percentages of Y variable counts')
p1 = dtf.groupby('Y').size() * 100 /len(dtf)
print(p1)


#Distplot for variable AGE
#numerical

bc2 = sns.displot(dtf.AGE.dropna(),kde=True)
bc2.set_titles('Distribution of age of passengers on the Titanic')
bc2.set_axis_labels('Age distribution','Count')
bc2.set_titles("dddd")

#Create a figure of a distplot and a boxplot for age
fig, ax = plt.subplots(1,2)
fig.suptitle('AGE')
ax[0].title.set_text('distribution')
sns.distplot(dtf['AGE'].dropna(),kde=True, ax=ax[0])
plt.xlabel('AGE')
plt.title('')

ax[1].title.set_text('Outliers')
tmp_dtf = pd.DataFrame(dtf['AGE'])
#tmp_dtf['AGE'] = np.log(tmp_dtf['AGE'])
tmp_dtf.boxplot(column='AGE', ax=ax[1])
plt.show    

#Bivariate plots
fig, ax = plt.subplots(nrows=1, ncols=3,  sharex=False, sharey=False, figsize=(8,6))
fig.suptitle('AGE   vs   Survival', fontsize=20)

            
### distribution
ax[0].title.set_text('density')
sns.histplot(dtf, x="AGE", multiple="stack", hue=dtf['Y'], element="step", kde=True, ax=ax[0])
ax[0].grid(True)

##cat plot of AGE
ax[1].title.set_text('outliers')
sns.boxplot(x=dtf['Y'], y=dtf['AGE'], data=dtf, whis=np.inf, ax=ax[1])
ax[1].set_xlabel('SURVIVAL')
ax[1].grid(True)


#create new variable AGEGROUP
dtf['AGEGROUP'] = pd.qcut(dtf['AGE'], 4, labels=['25%tile','50%tile','75%tile','100%tile'])
#show the counts for each group
print(dtf['AGE'].dtypes)
print(dtf['AGEGROUP'].dtypes)
c14= dtf['AGEGROUP'].value_counts(sort=False, dropna=True)
print(c14)

#crosstab AGEGROUP and SURVIVED
print(pd.crosstab(dtf['AGEGROUP'],dtf['Y']))

#plot the third chart to show survival for each bin
ax[2].title.set_text('bins')
ax[2] = sns.countplot(data=dtf, x=dtf['AGEGROUP'], hue=dtf['Y'], ax=ax[2])
ax[2].set_xticklabels(ax[2].get_xticklabels(), rotation=40,)
ax[2].grid(True)
plt.show()

#Test using ANOVA to see if the relationship is significant.

model = smf.ols('Y ~ AGE', data=dtf).fit()
print(model.summary())

#To make interpretation easy you can display the following:
table = sm.stats.anova_lm(model)
print(table)
p = table["PR(>F)"][0]
coeff, p = None, round(p, 3)
conclusion = "Correlated" if p < 0.05 else "Non-Correlated"
print("Anova F: the variables are", conclusion, "(p-value: "+str(p)+")")


#plot a bar chart to analyse Sex against Survival
x, y = "SEX", "Y"

#count the number of survivals per gender and then do it in percentage.
print(dtf.groupby(x)[y].value_counts())
print(dtf.groupby(x)[y].value_counts(normalize=True))
#count the number of survivals for each gender and then place the index in a list to use to order the chart.
order = dtf.groupby(x)[y].count().index.tolist()
sns.catplot(x=x, hue=y, data=dtf, kind='count', order=order)

#Test the correlation betwee the two categorical variables gender and survival.
#use chi-square to do this

cont_table = pd.crosstab(index=dtf[x], columns=dtf[y])
cont_table
scipy.stats.chi2_contingency(cont_table)


#The first value (260.717) is the Chi-square value, followed by the p-value(1.1973e-58), then comes the degrees of freedom 1), 
#and lastly it outputs the expected frequencies as an array. Since all of the expected frequencies are greater than 5, the chi2 test resutls can be trusted.
#We can reject the null hypothesis as the p-value is less than 0.05. Thus, the results indicate that there is a relationship between Gender 
#and survival.

## Create new column
dtf["CABIN_SECTION"] = dtf["CABIN"].apply(lambda x: str(x)[0])
print(dtf.head())

#Next we start to preprocess the dataset ready for our machine learning model.

#first we create a subset without the columns passengerid, name, ticket, cabin, and agegroup as non of these are to be used as predictors.
dtf = dtf[['Y','SEX','AGE','EMBARKED','CABIN_SECTION','PCLASS','FARE','SIBSP','PARCH']]

#partitioning the dataset into train and test sets
dtf_train, dtf_test = model_selection.train_test_split(dtf, 
                      test_size=0.3)
## print info
print("X_train shape:", dtf_train.drop("Y",axis=1).shape, "| X_test shape:", dtf_test.drop("Y",axis=1).shape)
print("y_train mean:", round(np.mean(dtf_train["Y"]),2), "| y_test mean:", round(np.mean(dtf_test["Y"]),2))
print(dtf_train.shape[1], "features:", dtf_train.drop("Y",axis=1).columns.to_list())

#replace missing data in AGE with the average of the age in the training and test set.
dtf_train.isnull().sum()
dtf_test.isnull().sum()
dtf_train.loc[dtf_train['AGE'].isnull(), ['AGE']]=dtf_train['AGE'].mean()
dtf_test.loc[dtf_test['AGE'].isnull(), ['AGE']]=dtf_test['AGE'].mean()
#show that the missing values in AGE have been replaced
dtf_train.isnull().sum()
dtf_test.isnull().sum()


##All further data preprocessing must be carried out on both the train set and the test set.

#train dataset
#encoding of categorical variable into a binary vector
## create dummy for SEX, EMBARKED, PCLASS and CABIN_SECTION
dummy = pd.get_dummies(dtf_train["SEX"], 
                       prefix="SEX",drop_first=True)
dtf_train= pd.concat([dtf_train, dummy], axis=1)
print( dtf_train.filter(like="SEX", axis=1).head() )


#create a dummy for EMBARKED
print(dtf['EMBARKED'].value_counts())
dummy = pd.get_dummies(dtf_train["EMBARKED"], 
                       prefix="EMBARKED",drop_first=True)
dtf_train= pd.concat([dtf_train, dummy], axis=1)
print( dtf_train.filter(like="EMBARKED", axis=1).head() )

#create a dummy for CABIN_SECTION
print(dtf['CABIN_SECTION'].value_counts())
dummy = pd.get_dummies(dtf_train["CABIN_SECTION"], 
                       prefix="CABIN_SECTION",drop_first=True)
dtf_train= pd.concat([dtf_train, dummy], axis=1)
print( dtf_train.filter(like="CABIN_SECTION", axis=1).head() )

#create a dummy for PCLASS
print(dtf['PCLASS'].value_counts())
dummy = pd.get_dummies(dtf_train["PCLASS"], 
                       prefix="PCLASS",drop_first=True)
dtf_train= pd.concat([dtf_train, dummy], axis=1)
print( dtf_train.filter(like="PCLASS", axis=1).head() )

## drop the original categorical columns
dtf_train.drop(columns=['SEX','CABIN_SECTION','EMBARKED','PCLASS'], inplace=True, axis=1)
print(dtf_train.head())



#Scaling the features
scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
X = scaler.fit_transform(dtf_train.drop(columns=["Y"], axis=1))
dtf_scaled= pd.DataFrame(X, columns=dtf_train.drop(columns=["Y"], axis=1).columns, index=dtf_train.index)
dtf_scaled["Y"] = dtf_train["Y"]
dtf_scaled.head()

#test dataset
#encoding of categorical variable into a binary vector
## create dummy for SEX, EMBARKED, PCLASS and CABIN_SECTION
dummy = pd.get_dummies(dtf_test["SEX"], 
                       prefix="SEX",drop_first=True)
dtf_test= pd.concat([dtf_test, dummy], axis=1)
print( dtf_test.filter(like="SEX", axis=1).head() )


#create a dummy for EMBARKED
print(dtf['EMBARKED'].value_counts())
dummy = pd.get_dummies(dtf_test["EMBARKED"], 
                       prefix="EMBARKED",drop_first=True)
dtf_test= pd.concat([dtf_test, dummy], axis=1)
print( dtf_test.filter(like="EMBARKED", axis=1).head() )

#create a dummy for CABIN_SECTION
print(dtf['CABIN_SECTION'].value_counts())
dummy = pd.get_dummies(dtf_test["CABIN_SECTION"], 
                       prefix="CABIN_SECTION",drop_first=True)
dtf_test= pd.concat([dtf_test, dummy], axis=1)
print( dtf_test.filter(like="CABIN_SECTION", axis=1).head() )

#create a dummy for PCLASS
print(dtf['PCLASS'].value_counts())
dummy = pd.get_dummies(dtf_test["PCLASS"], 
                       prefix="PCLASS",drop_first=True)
dtf_test= pd.concat([dtf_test, dummy], axis=1)
print( dtf_test.filter(like="PCLASS", axis=1).head() )

## drop the original categorical columns
dtf_test.drop(columns=['SEX','CABIN_SECTION','EMBARKED','PCLASS'], inplace=True, axis=1)
print(dtf_test.head())



#Scaling the features in train and test data sets
scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
X = scaler.fit_transform(dtf_train.drop(columns=["Y"], axis=1))
dtf_scaled= pd.DataFrame(X, columns=dtf_train.drop(columns=["Y"], axis=1).columns, index=dtf_train.index)
dtf_scaled["Y"] = dtf_train["Y"]
dtf_scaled.head()

scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
X = scaler.fit_transform(dtf_test.drop(columns=["Y"], axis=1))
dtf_scaled= pd.DataFrame(X, columns=dtf_test.drop(columns=["Y"], axis=1).columns, index=dtf_test.index)
dtf_scaled["Y"] = dtf_test["Y"]
dtf_scaled.head()

#Feature Selection using Lasso regularisation
X = dtf_train.drop("Y", axis=1).values
y = dtf_train["Y"].values
feature_names = dtf_train.drop("Y", axis=1).columns
## Anova
selector = feature_selection.SelectKBest(score_func=  
              feature_selection.f_classif, k=10).fit(X,y)
anova_selected_features = feature_names[selector.get_support()]

## Lasso regularization
selector = feature_selection.SelectFromModel(estimator= 
              linear_model.LogisticRegression(C=1, penalty="l1", 
              solver='liblinear'), max_features=10).fit(X,y)
lasso_selected_features = feature_names[selector.get_support()]
 
## Plot the barplot showing the feature selection
fig, ax = plt.subplots(nrows=1, ncols=1,  sharex=False, sharey=False, figsize=(8,6))
fig.suptitle('Lasso vs Anova', fontsize=20)
dtf_features = pd.DataFrame({"features":feature_names})
dtf_features["anova"] = dtf_features["features"].apply(lambda x: "anova" if x in anova_selected_features else "")
dtf_features["num1"] = dtf_features["features"].apply(lambda x: 1 if x in anova_selected_features else 0)
dtf_features["lasso"] = dtf_features["features"].apply(lambda x: "lasso" if x in lasso_selected_features else "")
dtf_features["num2"] = dtf_features["features"].apply(lambda x: 1 if x in lasso_selected_features else 0)
dtf_features["method"] = dtf_features[["anova","lasso"]].apply(lambda x: (x[0]+" "+x[1]).strip(), axis=1)
dtf_features["selection"] = dtf_features["num1"] + dtf_features["num2"]
sns.barplot(ax=ax, y="features", x="selection", hue="method", data=dtf_features.sort_values("selection", ascending=False), dodge=False)


#Ensemlbe method - Random Forest
X = dtf_train.drop("Y", axis=1).values
y = dtf_train["Y"].values
feature_names = dtf_train.drop("Y", axis=1).columns.tolist()

## Importance - random forests classifier 
model = ensemble.RandomForestClassifier(n_estimators=100,
                      criterion="entropy", random_state=0)
model.fit(X,y)
importances = model.feature_importances_


## Put in a pandas dtf, 2 variables in the data frame, IMPORTANCE and VARIABLE sorted by importance.
dtf_importances = pd.DataFrame({"IMPORTANCE":importances, 
            "VARIABLE":feature_names}).sort_values("IMPORTANCE", 
            ascending=False)
## cummulative sum
dtf_importances['cumsum'] =  dtf_importances['IMPORTANCE'].cumsum(axis=0)
##set the index to VARIABLE column
dtf_importances = dtf_importances.set_index("VARIABLE")
    
## Plot on a fig with two columns
fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=(8,6))
fig.suptitle("Features Importance", fontsize=20)

#first chart is a bar plotting variables by importance
ax[0].title.set_text('variables')
dtf_importances[["IMPORTANCE"]].sort_values(by="IMPORTANCE").plot(
                kind="barh", legend=False, ax=ax[0]).grid(axis="x")
ax[0].set(ylabel="")

##second chart is a line showing cumullative importance.
ax[1].title.set_text('cumulative')
dtf_importances[["cumsum"]].plot(kind="line", linewidth=4, 
                                 legend=False, ax=ax[1])
ax[1].set(xlabel="", xticks=np.arange(len(dtf_importances)), 
          xticklabels=dtf_importances.index)
plt.xticks(rotation=70)
plt.grid(axis='both')
plt.show()


#Select the following features, train and test
print(dtf_train.head())
X_names = ["AGE", "FARE", "SEX_male", "SIBSP", "PCLASS_3", "PARCH",
"CABIN_SECTION_n", "EMBARKED_S", "PCLASS_2", "CABIN_SECTION_F", "CABIN_SECTION_E", "CABIN_SECTION_D"]
X_train = dtf_train[X_names].values
y_train = dtf_train["Y"].values
X_test = dtf_test[X_names].values
y_test = dtf_test["Y"].values


#Designing the model requires figuring out what parameter settings to use
#RandomSearch helps to find the best combination of parameters and puts them into the model using the X_train and Y_train datasets
model = ensemble.GradientBoostingClassifier()
## define hyperparameters combinations to try
param_dic = {'learning_rate':[0.15,0.1,0.05,0.01,0.005,0.001],      #weighting factor for the corrections by new trees when added to the model
'n_estimators':[100,250,500,750,1000,1250,1500,1750],  #number of trees added to the model
'max_depth':[2,3,4,5,6,7],    #maximum depth of the tree
'min_samples_split':[2,4,6,8,10,20,40,60,100],    #sets the minimum number of samples to split
'min_samples_leaf':[1,3,5,7,9],     #the minimum number of samples to form a leaf
'max_features':[2,3,4,5,6,7],     #square root of features is usually a good starting point
'subsample':[0.7,0.75,0.8,0.85,0.9,0.95,1]}       #the fraction of samples to be used for fitting the individual base learners. Values lower than 1 generally lead to a reduction of variance and an increase in bias.
## random search
random_search = model_selection.RandomizedSearchCV(model, 
       param_distributions=param_dic, n_iter=1000, 
       scoring="accuracy").fit(X_train, y_train)
print("Best Model parameters:", random_search.best_params_)
print("Best Model mean accuracy:", random_search.best_score_)
model = random_search.best_estimator_

#use the model now
model.fit(X_train, y_train)
## test
predicted_prob = model.predict_proba(X_test)[:,1]
predicted = model.predict(X_test)


#Evaluation: AUC, Precision and Recall
## Accuray e AUC
accuracy = metrics.accuracy_score(y_test, predicted)
auc = metrics.roc_auc_score(y_test, predicted_prob)
print("Accuracy (overall correct predictions):",  round(accuracy,2))
print("Auc:", round(auc,2))
    
## Precision e Recall
recall = metrics.recall_score(y_test, predicted)
precision = metrics.precision_score(y_test, predicted)
print("Recall (all 1s predicted right):", round(recall,2))
print("Precision (confidence when predicting a 1):", round(precision,2))
print("Detail:")
print(metrics.classification_report(y_test, predicted, target_names=[str(i) for i in np.unique(y_test)]))

#confusion matrix
classes = np.unique(y_test)
fig, ax = plt.subplots()
cm = metrics.confusion_matrix(y_test, predicted, labels=classes)
sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False)
ax.set(xlabel="Pred", ylabel="True", title="Confusion matrix")
ax.set_yticklabels(labels=classes, rotation=0)
plt.show()

# test the prediction on random observation from the test set and see what is predicted.
print("True:", y_test[2], "--> Pred:", predicted[2], "| Prob:", np.max(predicted_prob[2]))
print("True:", y_test[3], "--> Pred:", predicted[3], "| Prob:", np.max(predicted_prob[3]))