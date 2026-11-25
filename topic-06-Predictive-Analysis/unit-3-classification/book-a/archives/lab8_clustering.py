# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 11:40:16 2018
Most recent update: 12/11/2024
@author: Brenda
"""

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os 

os.environ["OMP_NUM_THREADS"] = '1'
                   
"""
Data Management
"""
addhealth_data = pd.read_csv("lab08_addhealth.csv")

#set PANDAS to show all columns in Data frame
pd.set_option('display.max_columns', None)

#upper-case all DataFrame column names
addhealth_data.columns = map(str.upper, addhealth_data.columns)
addhealth_data.columns

# Data Management

var_in_use = [
    'ALCEVR1',
    'ALCPROBS1',
    'MAREVER1',
    'DEP1',
    'ESTEEM1',
    'VIOL1',
    'DEVIANT1',
    'FAMCONCT',
    'SCHCONN1',
    'PARACTV',
    'PARPRES'
    ]

var_validation = "GPA1"

 # Convert to numeric and coerce non-numeric values to NaN
for i in var_in_use + [var_validation]:
    addhealth_data[i] = addhealth_data[i].replace(" ", np.nan)
    addhealth_data[i] = pd.to_numeric(addhealth_data[i], errors='coerce')
    print('Descriptive stats for' +i)
    print(addhealth_data[i].describe())

 

# Drop rows that have missing values in any of the cluster variables or the validation variable
addhealth_data = addhealth_data.dropna(subset=var_in_use + [var_validation])


# subset clustering variables
cluster=addhealth_data[var_in_use].copy()
cluster.describe()


# Standardize the Clustering Variables
cluster = cluster.copy()
for v in var_in_use:
    cluster[v] = preprocessing.scale(cluster[v].astype("float64"))

#split data into train and test sets
clus_train, clus_test = train_test_split(cluster, test_size=.3, random_state=123)


# run the cluster analysis to calculate the total average distance from observations to centroid for 1-9 clusters
meandist = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(clus_train)
    meandist.append(kmeans.inertia_)  # Sum of squared distances to nearest cluster center


#plot the graph
plt.plot(range(1,10), meandist)
plt.xlabel('Number of clusters')
plt.ylabel('Average distance')
plt.title('Selecting k with the Elbow Method')

#choose the optimum number of clusters
optimum_clusters = 3


# Perform final clustering with selected number of clusters
kmeans_final = KMeans(n_clusters=optimum_clusters, random_state=42, n_init='auto')
kmeans_final.fit(clus_train)
cluster_labels = kmeans_final.labels_


#plot the graph using PCA to create condensed version so it plots using two dimensions Canonical variables 1 and 2 
from sklearn.decomposition import PCA
pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(clus_train)
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=cluster_labels,)
plt.xlabel('Canonical variable 1')
plt.ylabel('Canonical variable 2')
plt.title('Scatterplot of Canonical Variables for 3 Clusters')
plt.show()



#create a unique identifier variable from the index for the
#cluster training data to merge with the cluster assignment variable

#When we reset the index, the old index is added as a column, and a new sequential index #is used. 
#You can see the new column is called index.
clus_train.reset_index(level=0, inplace=True)
clus_train.head()

#create a list that holds the new index variable.
cluslist=list(clus_train['index'])
print(cluslist)

#create a list of cluster assignments
labels=list(kmeans_final.labels_)
print(labels)

#combine index variable list with cluster assignment list into a dictionary
newlist=dict(zip(cluslist, labels))
newlist

#convert newlist dictionary to a dataframe
newclus=DataFrame.from_dict(newlist, orient='index')
newclus.head(25)


#rename the cluster assignment column
newclus.columns=['cluster']

#create a unique identifier variable from the index for the
#cluster assignment dataframe
#to merge with cluster training data
newclus.reset_index(level=0, inplace=True)
newclus.head()

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


#We first have to extract the GPA variable from the original data set, 
gpa_data = addhealth_data['GPA1']
gpa_data.head()
#split the data into test and training datasets, 
gpa_train, gpa_test = train_test_split(gpa_data, test_size=0.3, random_state=123)
gpa_train1=DataFrame(gpa_train)

# create a new index on the dataframe
gpa_train1.reset_index(level=0, inplace=True)
gpa_train1.head()

#merge the gpa training observations in the dataset that includes the 
#cluster assignment variable with the training observations in the 
#dataset that includes the clustering variables.
merged_train_all=pd.merge(gpa_train1, merged_train, on='index')
merged_train_all.head()

#we  create a new dataframe that only contains the two columns GPA1 and cluster
#with any rows that have NAN's removed.

sub1=merged_train_all[['GPA1','cluster']].dropna()
sub1.head()

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


# Then, because our categorical cluster variable has three categories (0,1 and 2),
# we will request a tukey test to evaluate post hot comparisons between the clusters
# using the multi comparison function from the statsmodels.stats.multicomplibrary
# which we imported as multi.

mc1= multi.MultiComparison(sub1['GPA1'],sub1['cluster'])
res1=mc1.tukeyhsd()
print(res1.summary())

# Here are the results.
#The analysis of variance summary table indicates that the clusters differed
#significantly on GPA. When we examine the means, we find that not surprisingly,
#adolescents in cluster 0, the most troubled group, had the lowest GPA, and
#adolescents in cluster 2, the least troubled group, had the highest GPA.
#The tukey test shows that the clusters differed significantly in mean GPA




#work on cluster assignment with clustering variables to examine means by cluster.

#uses code from student, but doesn't do the full job
# Create results dataframe with cluster assignments and income data
#results_df = pd.DataFrame({
 #   'Cluster': cluster_labels,
  #  'incomeperperson': df.iloc[train_idx][validation_var].values  # Use iloc with saved indices
#})
# Analyze cluster characteristics
#cluster_means = df.iloc[train_idx][cluster_vars + [validation_var]].copy()
#cluster_means['Cluster'] = cluster_labels
#cluster_summary = cluster_means.groupby('Cluster').mean()

#print("Cluster Means:")
#print(cluster_summary)

