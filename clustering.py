#Call required libraries
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation #For clustering
from sklearn.mixture import GaussianMixture #For GMM clustering
import os                     # For os related operations
import sys                    # For data size
import time                   # To time processes
import warnings               # To suppress warnings
import random
import numpy as np            # Data manipulation
import pandas as pd           # Dataframe manipulatio 
import matplotlib.pyplot as plt                   # For graphics
from sklearn import cluster, mixture
#import seaborn as sns
data = pd.read_csv('clustering_data.csv')
data_plot=data['Profit']
random.seed(10)
#dropping columns
#data1=data
#data=(data.drop['a'],axis=0, inplace = True)
#print(data.drop(data.columns[0],axis=1))
#data.drop(['Customer.ID'], axis=1,inplace = True)
#print(data.drop(['a'], axis=1))
data.drop(['Country'], axis=1, inplace = True)
data1=data
#mean and median of profit
print(np.median(data['Profit']))
print(np.mean(data['Profit']))
#labelling function based on profit
def func(x):
    if x < -500:
        return 0
    elif -500<x < 0:
        return 1
    elif 0<x <500:
        return 2
    else:
        return 3
data2=data   
y_test = data['Profit'].apply(func)
#if(data[data['Profit'] >= np.median(data['Profit'])]):	
clmns = ['Customer.Name', 'Customer Type']
clmns1=['Customer.Name', 'Customer Type','Profit','Sales','Shipping.Cost','Quantity']
#categorical features Encoding
data["Customer.Name"] = data["Customer.Name"].astype('category')
data["Customer.Name"] = data["Customer.Name"].cat.codes
data["Customer Type"] = data["Customer Type"].astype('category')
data["Customer Type"] = data["Customer Type"].cat.codes
data["Customer.ID"] = data["Customer.ID"].astype('category')
data["Customer.ID"] = data["Customer.ID"].cat.codes
data2["Customer.Name"] = data1["Customer.Name"].astype('category')
data2["Customer.Name"] = data1["Customer.Name"].cat.codes
data2["Customer Type"] = data1["Customer Type"].astype('category')
data2["Customer Type"] = data1["Customer Type"].cat.codes
data2["Customer.ID"] = data2["Customer.ID"].astype('category')
data2["Customer.ID"] = data2["Customer.ID"].cat.codes
data_plot1=data['Customer Type']
data_plot2=data['Customer.Name']
data_plot3=data['Customer.ID']
#data2.drop(['Profit'], axis=1,inplace = True)
#data2.drop(['Sales'], axis=1,inplace = True)
#data2.drop(['Quantity'], axis=1,inplace = True)
data2.drop(['Shipping.Cost'], axis=1,inplace = True)
#data2.drop(['Customer.Name'], axis=1,inplace = True)
data2.drop(['Customer Type'], axis=1,inplace = True)
from sklearn import preprocessing
#normalizing the data
def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result
data=normalize(data)
data2=normalize(data2)
k_means = KMeans(n_clusters=4)
k_means.fit(data2)
y_pred=(k_means.predict(data2))
#Agglomerative Clustering
def doAgglomerative(X, nclust=4):
    model = AgglomerativeClustering(n_clusters=nclust, affinity = 'euclidean', linkage = 'ward')
    clust_labels1 = model.fit_predict(X)
    return (clust_labels1)
#y_pred = doAgglomerative(data2, 4)
#Affinity Clustering
def doAffinity(X):
    model = AffinityPropagation(damping = 0.5, max_iter = 2, affinity = 'euclidean')
    model.fit(X)
    clust_labels2 = model.predict(X)
    cent2 = model.cluster_centers_
    return (clust_labels2, cent2)
#y_pred, cent2 = doAffinity(data2)
#GaussianMixture
def doGMM(X, nclust=4):
    model = GaussianMixture(n_components=nclust,init_params='kmeans')
    model.fit(X)
    clust_labels3 = model.predict(X)
    return (clust_labels3)
#y_pred = doGMM(data2,4)
def MeanShift(x,y):
    ms=cluster.MeanShift(x)
    ms_result=ms.fit_predict(y)
    return(ms_result)
#y_pred=MeanShift(0.1,data2)
def MiniKmeans(x, y):
    mb= cluster.MiniBatchKMeans(x)
    mb_result=mb.fit_predict(y)
    return(mb_result)
#y_pred = MiniKmeans(4,data)
spectral = cluster.SpectralClustering(n_clusters=4)
#y_pred= spectral.fit_predict(data2)
def Dbscan(x, y):
    db=cluster.DBSCAN(eps=x)
    db_result=db.fit_predict(y)
    return(db_result)
#y_pred = Dbscan(0.3,data2)
def Affinity(x, y,z):
    ap=cluster.AffinityPropagation(damping=x, preference=y)
    ap_result=ap.fit_predict(z)
    return(ap_result)
#y_pred = Affinity(0.9,-200,data2)
#Birch Clustering
def Bir(x, y):
    bi=cluster.Birch(n_clusters=x)
    bi_result=bi.fit_predict(y)
    return(bi_result)
#y_pred= Bir(4,data2)
#Ensemble K-means clustering
def ensemble_kmeans(data, rnd_states, k_list):    
    labs=[]    
    clusterer = KMeans(n_clusters=4, random_state=0)
    clusterer.fit(data)
    labs.append(clusterer.labels_)
    return  pd.DataFrame(labs)
#y_pred=ensemble_kmeans(data2, 0, 4)
#accuracy_score
from sklearn.metrics import accuracy_score #works
score = accuracy_score(y_test,y_pred)
print('Accuracy:{0:f}'.format(score))
#plot
fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(data_plot1,data_plot,
                     c=y_pred,s=50)
ax.set_title('Data Visualization')
ax.set_xlabel('Customer Type')
ax.set_ylabel('Profit')
plt.colorbar(scatter)
plt.show()


