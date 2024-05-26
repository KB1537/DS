import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style
from sklearn.cluster import KMeans

df=pd.read_csv('clustering_data.csv')
df_mapped=df.copy()
df_mapped['Language']= df_mapped['Language'].map({'English':0,'French':1,'German':2}) #NOT optimal way to encode data 

x= df_mapped.iloc[:,1:4]


Kmeans = KMeans(2)
Kmeans.fit(x)

identified_clusters = Kmeans.fit_predict(x)
print(identified_clusters)

df_with_clusters = df.copy()
df_with_clusters['clusters'] = identified_clusters




plt.scatter(df_with_clusters['Longitude'],df_with_clusters['Latitude'],c=df_with_clusters['clusters'],cmap='rainbow')
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show()



print(Kmeans.inertia_)

#calculate WCSS
wcss=[]
for i in range(1,7):#no. of observations +1 
    Kmeans= KMeans(i)
    Kmeans.fit(x)
    wcss_iter=Kmeans.inertia_
    wcss.append(wcss_iter)

print(wcss)


# plot elbow method chart
number_clusters = range(1,7)
plt.plot(number_clusters,wcss)
plt.title('the elbow method')
plt.xlabel('number of clusters')
plt.ylabel('Within-cluster sum of squars')
plt.show()