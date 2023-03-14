# %%
# Imports

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
# %%
df_allPackets_train = pd.read_csv("train_data.csv",names=["PacketType", "PacketLength"])
df_allPackets_test = pd.read_csv("test_data.csv",names=["PacketType", "PacketLength"])

df_allPackets_train = df_allPackets_train.replace({'Chat': 0, 'Audio': 1, 'Email': 2,'FTPS': 3,'Video': 4})
df_allPackets_test = df_allPackets_test.replace({'Chat': 0, 'Audio': 1, 'Email': 2,'FTPS': 3,'Video': 4})

df_allPackets = pd.concat([df_allPackets_train,df_allPackets_test], axis=0, sort=False)
#df_allPackets = df_allPackets.replace({'Chat': 0, 'Audio': 1, 'Email': 2,'FTPS': 3,'Video': 4})
# %%


trainLengthData = df_allPackets_train['PacketLength'].to_numpy().reshape(-1, 1).astype('int')
trainType = df_allPackets_train['PacketType'].to_numpy().astype('int')

testLengthData = df_allPackets['PacketLength'].to_numpy().reshape(-1, 1).astype('int')
testType = df_allPackets['PacketType'].to_numpy().astype('int')

finalLengthData = df_allPackets['PacketLength'].to_numpy().reshape(-1, 1).astype('int')
finalType = df_allPackets['PacketType'].to_numpy().astype('int')

print(finalLengthData.dtype)

kmeans = KMeans(n_clusters=5,init="random")
kmeans = kmeans.fit(finalLengthData)

print(kmeans.labels_)

finalPredicted = kmeans.labels_

plt.hist(finalLengthData,200)

# %%


import seaborn as sns

x = finalLengthData.flatten()
y = finalType.flatten()

n_bins = 200  # number of bins for the histogram

df = pd.DataFrame({'x': x, 'y': y})

_, bin_edges = np.histogram(x, bins=n_bins)
df['bin'] = pd.cut(x, bins=bin_edges, labels=False, include_lowest=True)

color = df.groupby('bin').mean()['y']
#df['color'] = df.bin.apply(lambda k: color[k])
df['Categoría'] = y

df

p = sns.histplot(data=df, x='x', bins=bin_edges, hue='Categoría', palette='tab10');
p.set_xlabel("Longitud del paquete (bytes)")
p.set_ylabel("Histograma de paquetes por \n categoría y categoría real (color)")
plt.legend(labels=["Video","FTPS","Email","Audio","Chat"])
plt.savefig('CategoriasHistograma.pdf',bbox_inches='tight')  
# %%

df = pd.DataFrame({'x': x, 'y': y})

_, bin_edges = np.histogram(x, bins=n_bins)
df['bin'] = pd.cut(x, bins=bin_edges, labels=False, include_lowest=True)

color = df.groupby('bin').mean()['y']
#df['color'] = df.bin.apply(lambda k: color[k])
df["Categoría"] = finalPredicted

p = sns.histplot(data=df, x='x', bins=bin_edges, hue='Categoría', palette='tab10');
p.set_xlabel("Longitud del paquete (bytes)")
p.set_ylabel("Histograma de paquetes por categoría y categoría \npredicha por k-Means (color) con 5 grupos")
plt.legend(labels=["Categoría 1","Categoría 2","Categoría 3","Categoría 4","Categoría 5"])
plt.savefig('k-means5.pdf',bbox_inches='tight') 
# %%

kmeans = KMeans(n_clusters=8)
kmeans = kmeans.fit(finalLengthData)

print(kmeans.labels_)

finalPredicted = kmeans.labels_

df = pd.DataFrame({'x': x, 'y': y})

_, bin_edges = np.histogram(x, bins=n_bins)
df['bin'] = pd.cut(x, bins=bin_edges, labels=False, include_lowest=True)

color = df.groupby('bin').mean()['y']
#df['color'] = df.bin.apply(lambda k: color[k])
df["Categoría"] = finalPredicted

p = sns.histplot(data=df, x='x', bins=bin_edges, hue='Categoría', palette='tab10');
p.set_xlabel("Longitud del paquete (bytes)")
p.set_ylabel("Histograma de paquetes por categoría y categoría \npredicha por k-Means (color) con 8 grupos")
plt.legend(labels=["Categoría 1","Categoría 2","Categoría 3","Categoría 4","Categoría 5","Categoría 6","Categoría 7","Categoría 8"])
plt.savefig('k-means8.pdf',bbox_inches='tight')  

# %%

from sklearn.cluster import DBSCAN

clustering = DBSCAN(eps=1, min_samples=20).fit(finalLengthData)
finalPredicted = clustering.labels_

df = pd.DataFrame({'x': x, 'y': y})

_, bin_edges = np.histogram(x, bins=n_bins)
df['bin'] = pd.cut(x, bins=bin_edges, labels=False, include_lowest=True)

color = df.groupby('bin').mean()['y']
#df['color'] = df.bin.apply(lambda k: color[k])
df["Categoría"] = finalPredicted
df.loc[df["Categoría"]==-1, "Categoría"] = 4
p = sns.histplot(data=df, x='x', bins=bin_edges, hue='Categoría', palette='tab10');
p.set_xlabel("Longitud del paquete (bytes)")
p.set_ylabel("Histograma de paquetes por categoría y categoría \npredicha por DBSCAN (color) con 5 grupos encontrados")
plt.legend(labels=["Categoría 1","Categoría 2","Categoría 3","Categoría 4","Categoría 5"])
plt.savefig('DBSCAN.pdf',bbox_inches='tight')  


#clf.predict_proba([[150]])
# %%
#tree.plot_tree(clf,filled=True)
#plt.savefig('tree1.pdf',format='pdf',bbox_inches = "tight")
# %%

#clf = tree.DecisionTreeClassifier(max_depth=3)
#clf = clf.fit(testLengthData, testType)

#predictedTestType = clf.predict(testLengthData)
#acc = accuracy_score(testType, predictedTestType)

#print(acc)
# %%