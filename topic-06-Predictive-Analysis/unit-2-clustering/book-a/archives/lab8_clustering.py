# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 11:40:16 2018

@author: Brenda
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
data = pd.read_csv("lab08_addhealth.csv")

#set PANDAS to show all columns in Data frame
pd.set_option('display.max_columns', None)

#upper-case all DataFrame column names
data.columns = map(str.upper, data.columns)
data.columns
# Data Management


data['ALCEVR1']=data['ALCEVR1'].replace(" ", np.NaN)
data['ALCEVR1'] = pd.to_numeric(data['ALCEVR1'])
data['ALCPROBS1']=data['ALCPROBS1'].replace(" ", np.NaN)
data['ALCPROBS1'] = pd.to_numeric(data['ALCPROBS1'])
data['MAREVER1']=data['MAREVER1'].replace(" ", np.NaN)
data['MAREVER1'] = pd.to_numeric(data['MAREVER1'])
data['DEP1']=data['DEP1'].replace(" ", np.NaN)
data['DEP1'] = pd.to_numeric(data['DEP1'])
data['ESTEEM1']=data['ESTEEM1'].replace(" ", np.NaN)
data['ESTEEM1'] = pd.to_numeric(data['ESTEEM1'])
data['VIOL1']=data['VIOL1'].replace(" ", np.NaN)
data['VIOL1'] = pd.to_numeric(data['VIOL1'])
data['DEVIANT1']=data['DEVIANT1'].replace(" ", np.NaN)
data['DEVIANT1'] = pd.to_numeric(data['DEVIANT1'])
data['FAMCONCT']=data['FAMCONCT'].replace(" ", np.NaN)
data['FAMCONCT'] = pd.to_numeric(data['FAMCONCT'])
data['SCHCONN1']=data['SCHCONN1'].replace(" ", np.NaN)
data['SCHCONN1'] = pd.to_numeric(data['SCHCONN1'])
data['PARACTV']=data['PARACTV'].replace(" ", np.NaN)
data['PARACTV'] = pd.to_numeric(data['PARACTV'])
data['PARPRES']=data['PARPRES'].replace(" ", np.NaN)
data['PARPRES'] = pd.to_numeric(data['PARPRES'])

data_clean = data.dropna()

# subset clustering variables
cluster=data_clean[['ALCEVR1','MAREVER1','ALCPROBS1','DEVIANT1','VIOL1',
'DEP1','ESTEEM1','SCHCONN1','PARACTV', 'PARPRES','FAMCONCT']]
cluster.describe()

# standardize clustering variables to have mean=0 and sd=1
clustervar=cluster.copy()
clustervar['ALCEVR1']=preprocessing.scale(clustervar['ALCEVR1'].astype('float64'))
clustervar['ALCPROBS1']=preprocessing.scale(clustervar['ALCPROBS1'].astype('float64'))
clustervar['MAREVER1']=preprocessing.scale(clustervar['MAREVER1'].astype('float64'))
clustervar['DEP1']=preprocessing.scale(clustervar['DEP1'].astype('float64'))
clustervar['ESTEEM1']=preprocessing.scale(clustervar['ESTEEM1'].astype('float64'))
clustervar['VIOL1']=preprocessing.scale(clustervar['VIOL1'].astype('float64'))
clustervar['DEVIANT1']=preprocessing.scale(clustervar['DEVIANT1'].astype('float64'))
clustervar['FAMCONCT']=preprocessing.scale(clustervar['FAMCONCT'].astype('float64'))
clustervar['SCHCONN1']=preprocessing.scale(clustervar['SCHCONN1'].astype('float64'))
clustervar['PARACTV']=preprocessing.scale(clustervar['PARACTV'].astype('float64'))
clustervar['PARPRES']=preprocessing.scale(clustervar['PARPRES'].astype('float64'))

#split data into train and test sets
clus_train, clus_test = train_test_split(clustervar, test_size=.3, random_state=123)

# k-means cluster analysis for 1-9 clusters
from scipy.spatial.distance import cdist
clusters=range(1,10)
meandist=[]

#run the cluster analysis and calculate the total average distance from observations to the centroid
#for each number of clusters so we can then plot it on the elbow graph
for k in clusters:
    model=KMeans(n_clusters=k,  n_init='auto')
    model.fit(clus_train)
    clusassign=model.predict(clus_train)
    meandist.append(sum(np.min(cdist(clus_train, model.cluster_centers_, 'euclidean'), axis=1))
    / clus_train.shape[0])


#plot the graph
plt.plot(clusters, meandist)
plt.xlabel('Number of clusters')
plt.ylabel('Average distance')
plt.title('Selecting k with the Elbow Method')

#now run it for 3 clusters
model3=KMeans(n_clusters=3,  n_init='auto')
model3.fit(clus_train)
clusassign=model3.predict(clus_train)

#plot the graph
from sklearn.decomposition import PCA
pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(clus_train)
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=model3.labels_,)
plt.xlabel('Canonical variable 1')
plt.ylabel('Canonical variable 2')
plt.title('Scatterplot of Canonical Variables for 3 Clusters')
plt.show()


#work on cluster assignment with clustering variables to examine means by cluster.

#create a unique identifier variable from the index for the 
#cluster training data to merge with the cluster assignment variable
clus_train.reset_index(level=0, inplace=True)

#create a list that has the new index variable.
cluslist=list(clus_train['index'])
cluslist

#create a list of cluster assignments
labels=list(model3.labels_)
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

#merge cluster assignment dataframe with te cluster tainning variable dataframe#
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

#Cluster 2 shows .64 for ALCEVR1 showing a high likelihood of having used alcohol 
#and 1.06 for use of Maijuana, they also have the highest levels of deviant behviour, depression, 
#and low self-esteem, and the lowest levels of school connectedness, parental presence, and family connectedness.

#Cluster 0 shows a low likelihood of using alcohol at -1.05 and low likelihood of 
#using marijuana at -0.47. They had lowest levels in depression, deviant behaviour
# and violence and highest levels of self esteem, family connectedness, parental 
#presence and school connectedness.



#We first have to extract the GPA variable from the original data set, 
#split the data into test and training datasets, and then 
#merge the training observations in the dataset that includes the cluster assignment 
#variable with the training observations in the 
#dataset that includes the clustering variables.


gpa_data = data['GPA1']

gpa_train, gpa_test = train_test_split(gpa_data, test_size=0.3, random_state=123)
gpa_train1=DataFrame(gpa_train)
gpa_train1.reset_index(level=0, inplace=True)
merged_train_all=pd.merge(gpa_train1, merged_train, on='index')

#we  create a new dataframe that only contains the two columns GPA1 and cluster
#with any rows that have NAN's removed.

sub1=merged_train_all[['GPA1','cluster']].dropna()
#We'll use analysis of variance to test whether there are significant differences 
#between clusters on the quantitative GPA variable.

#first we have to import in the statsmodels.formula.api and the 
#statsmodels.stats.multicomp libraries. We use the ols function to test the
# analysis of variance (ANOVA).

import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi

#remember the response variable goes first and then any catagory variables are 
#noted by the 'C'
gpamod = smf.ols(formula='GPA1 ~ C(cluster)',data=sub1).fit()
print(gpamod.summary())

#We can also print the mean GPA and standard deviation for each cluster using the 
#groupby function.

print('means by GPA by cluster')
m1=sub1.groupby('cluster').mean()
print(m1)

print('std deviations by cluster')
m2=sub1.groupby('cluster').std()
print(m2)


# Then, because our categorical cluster variable has three categories,
# we will request a tukey test to evaluate post hot comparisons between the clusters 
#using the multi comparison function from the statsmodels.stats.multicomplibrary 
#which we imported as multi.

mc1= multi.MultiComparison(sub1['GPA1'],sub1['cluster']) 
res1=mc1.tukeyhsd()
print(res1.summary())

# Here are the results. 
#The analysis of variance summary table indicates that the clusters differed 
#significantly on GPA. When we examine the means, we find that not surprisingly, 
#adolescents in cluster 1, the most troubled group, had the lowest GPA, and 
#adolescents in cluster 2, the least troubled group, had the highest GPA. 
#The tukey test shows that the clusters differed significantly in mean GPA,
# although the difference between cluster 0 and cluster 2 were smaller.



