import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import preprocessing
sns.set_theme

df = pd.read_csv('market_data.csv')

plt.scatter(df['Satisfaction'], df['Loyalty'])
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')
plt.show()

x = df.copy()

Kmeans = KMeans(2)
Kmeans.fit(x)

#clustering results

clusters = x.copy()
clusters['cluster_pred']=Kmeans.fit_predict(x)

plt.scatter(clusters['Satisfaction'],clusters['Loyalty'],c=clusters['cluster_pred'],cmap='rainbow')
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')
plt.show()

#standerdise variables  
x_scaled = preprocessing.scale(x)

#elbow methoid chart 
wcss=[]
for i in range(1,10):
    Kmeans=KMeans(i)
    Kmeans.fit(x_scaled)
    wcss.append(Kmeans.inertia_)

print(wcss)

plt.plot(range(1,10),wcss)
plt.xlabel('number of clusters')
plt.ylabel('WCSS')
plt.show()

Kmeans_new=KMeans(4)
Kmeans_new.fit(x_scaled)
clusters_new=x.copy()
clusters_new['cluster_pred']=Kmeans_new.fit_predict(x_scaled)

plt.scatter(clusters_new['Satisfaction'],clusters_new['Loyalty'],c=clusters_new['cluster_pred'],cmap='rainbow')
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')
plt.show()