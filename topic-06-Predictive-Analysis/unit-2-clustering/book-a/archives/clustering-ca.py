# -*- coding: utf-8 -*-
"""
Created on Fri May  4 12:17:00 2018

@author: BMULLALLY
"""

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans
import os 

os.environ["OMP_NUM_THREADS"] = '1'
   

"""
Data Management
"""
data = pd.read_csv("gapminder.csv")

#upper-case all DataFrame column names
data.columns = map(str.upper, data.columns)

# Data Management


data['INCOMEPERPERSON']=data['INCOMEPERPERSON'].replace(" ", np.NaN)
data['INCOMEPERPERSON'] = pd.to_numeric(data['INCOMEPERPERSON'])
data['FEMALEEMPLOYRATE']=data['FEMALEEMPLOYRATE'].replace(" ", np.NaN)
data['FEMALEEMPLOYRATE'] = pd.to_numeric(data['FEMALEEMPLOYRATE'])
data['INTERNETUSERATE']=data['INTERNETUSERATE'].replace(" ", np.NaN)
data['INTERNETUSERATE'] = pd.to_numeric(data['INTERNETUSERATE'])
data['LIFEEXPECTANCY']=data['LIFEEXPECTANCY'].replace(" ", np.NaN)
data['LIFEEXPECTANCY'] = pd.to_numeric(data['LIFEEXPECTANCY'])
data['ALCCONSUMPTION']=data['ALCCONSUMPTION'].replace(" ", np.NaN)
data['ALCCONSUMPTION'] = pd.to_numeric(data['ALCCONSUMPTION'])
data['URBANRATE']=data['URBANRATE'].replace(" ", np.NaN)
data['URBANRATE'] = pd.to_numeric(data['URBANRATE'])


data_clean = data.dropna()


# subset clustering variables
cluster=data_clean[['FEMALEEMPLOYRATE','INTERNETUSERATE','LIFEEXPECTANCY',
'ALCCONSUMPTION','URBANRATE']]
cluster.describe()


# standardize clustering variables to have mean=0 and sd=1
clustervar=cluster.copy()
clustervar['FEMALEEMPLOYRATE']=preprocessing.scale(clustervar['FEMALEEMPLOYRATE'].astype('float64'))
clustervar['INTERNETUSERATE']=preprocessing.scale(clustervar['INTERNETUSERATE'].astype('float64'))
clustervar['LIFEEXPECTANCY']=preprocessing.scale(clustervar['LIFEEXPECTANCY'].astype('float64'))
clustervar['ALCCONSUMPTION']=preprocessing.scale(clustervar['ALCCONSUMPTION'].astype('float64'))
clustervar['URBANRATE']=preprocessing.scale(clustervar['URBANRATE'].astype('float64'))

clus_train, clus_test = train_test_split(clustervar, test_size=.3, random_state=123)

from scipy.spatial.distance import cdist
clusters=range(1,10)
meandist=[]
for k in clusters:
    model=KMeans(n_clusters=k, n_init='auto')
    model.fit(clus_train)
    clusassign=model.predict(clus_train)
    meandist.append(sum(np.min(cdist(clus_train, model.cluster_centers_, 'euclidean'), axis=1))
    / clus_train.shape[0])
    

plt.plot(clusters, meandist)
plt.xlabel('Number of clusters')
plt.ylabel('Average distance')
plt.title('Selecting k with the Elbow Method')

#run for 2 clusters only
model2=KMeans(n_clusters=2, n_init='auto')
model2.fit(clus_train)
clusassign=model2.predict(clus_train)

from sklearn.decomposition import PCA
pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(clus_train)
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=model2.labels_,)
plt.xlabel('Canonical variable 1')
plt.ylabel('Canonical variable 2')
plt.title('Scatterplot of Canonical Variables for 2 Clusters')
plt.show()

#run for 4 clusters
model4=KMeans(n_clusters=4,  n_init='auto')
model4.fit(clus_train)
clusassign=model4.predict(clus_train)
from sklearn.decomposition import PCA
pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(clus_train)
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=model4.labels_,)
plt.xlabel('Canonical variable 1')
plt.ylabel('Canonical variable 2')
plt.title('Scatterplot of Canonical Variables for 4 Clusters')
plt.show()

#work on cluster assignment with clustering variables to examine means by cluster.

#create a unique identifier variable from the index for the 
#cluster training data to merge with the cluster assignment variable
clus_train.reset_index(level=0, inplace=True)

#create a list that has the new index variable.
cluslist=list(clus_train['index'])
cluslist

#create a list of cluster assignments
labels=list(model2.labels_)
labels

#combine index variable list with cluster assignment list into a dictionary
newlist=dict(zip(cluslist, labels))
newlist

#convert newlist dictionary to a dataframe
newclus=DataFrame.from_dict(newlist, orient='index')
newclus.head(25)


#now do the same for the cluster assignment variable 
#rename the cluster assignment column
newclus.columns=['cluster']

#create a unique identifier variable from the index for the
#cluster assignment dataframe
#to merge with cluster training data
newclus.reset_index(level=0, inplace=True)

#merge cluster assignment dataframe with the cluster training variable dataframe#
#by the index variable

merged_train=pd.merge(clus_train, newclus, on='index')
merged_train.head(20)

#cluster frequencies written two different ways
merged_train['cluster'].value_counts()
merged_train.cluster.value_counts()

#finally calculate clustering variable means by cluster
clustergrp = merged_train.groupby('cluster').mean()
print('Clustering variable means by cluster')
print(clustergrp)

#Cluster 0 shows low scores for female employement rate, internet use rate, life expectancy, 
#urban rate and alcohol consumption


#Cluster 1 shows higher scores for all variables compared to cluster 0 



#We first have to extract the income per person from the original data set, 
income_data = data['INCOMEPERPERSON']

income_train, income_test = train_test_split(income_data, test_size=0.3, random_state=123)
income_train1=DataFrame(income_train)
income_train1.reset_index(level=0, inplace=True)
merged_train_all=pd.merge(income_train1, merged_train, on='index')

#we  create a new dataframe that only contains the two columns incomeperperson and cluster
#with any rows that have NAN's removed.

sub1=merged_train_all[['INCOMEPERPERSON','cluster']].dropna()

#We'll use analysis of variance to test whether there are significant differences 
#between clusters on the quantitative GPA variable.

#first we have to import in the statsmodels.formula.api and the 
#statsmodels.stats.multicomp libraries. We use the ols function to test the
# analysis of variance (ANOVA).

import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi

#remember the response variable goes first and then any catagory variables are 
#noted by the 'C'
gpamod = smf.ols(formula='INCOMEPERPERSON ~ C(cluster)',data=sub1).fit()
print(gpamod.summary())

#We can also print the mean GPA and standard deviation for each cluster using the 
#groupby function.

print('means by INCOMEPERPERSON by cluster')
m1=sub1.groupby('cluster').mean()
print(m1)

print('std deviations by cluster')
m2=sub1.groupby('cluster').std()
print(m2)


# Here are the results. 
#The analysis of variance summary table indicates that the clusters differed 
#significantly on INCOMEPERPERON. When we examine the means, we find that not surprisingly, 
#countries in cluster 0,  had the lowest INCOMPERPERSON, and 
#countries in cluster 1, had the highest INCOMEPERPERON. 



