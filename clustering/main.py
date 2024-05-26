import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style
from sklearn.cluster import KMeans

df=pd.read_csv('clustering_data.csv')

x = df.iloc[:,1:3]
Kmeans = KMeans(3)

print(Kmeans.fit(x))

identified_clusters = Kmeans.fit_predict(x)
print(identified_clusters)

df_with_clusters = df.copy()
df_with_clusters['clusters'] = identified_clusters
print(df_with_clusters)

df_mapped=df.copy()
df_mapped['language']= df_mapped['language'].map({'English':0,'French':1,'German':2}) #NOT optimal way to encode data 


plt.scatter(df_with_clusters['Longitude'],df_with_clusters['Latitude'],c=df_with_clusters['clusters'],cmap='rainbow')
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show()