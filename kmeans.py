#K_Means

#Cargar Librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Cargar Dataset
dataset = pd.read_csv('Mall_Customers.csv')
X =  dataset.iloc[ : ,[3,4]].values

#Metodo del Codo
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = "k-means++", max_iter = 300, n_init = 10, random_state= 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title("Metodo del Codo")
plt.xlabel("Numero de Clusters")
plt.ylabel("WCSS(K)")
plt.show()


#Aplicar Metodo K-Means
kmeans = KMeans(n_clusters = 5, init = "k-means++", max_iter=300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)

#Representacion Graficas 
plt.scatter(X[y_kmeans ==0, 0], X[y_kmeans == 0,1], s =100, c = "red", label = "Cautos")
plt.scatter(X[y_kmeans ==1, 0], X[y_kmeans == 1,1], s =100, c = "blue",label = "Estandar")
plt.scatter(X[y_kmeans ==2, 0], X[y_kmeans == 2,1], s =100, c = "green",label = "Objetivo")
plt.scatter(X[y_kmeans ==3, 0], X[y_kmeans == 3,1], s =100, c = "cyan", label = "Descuidados")
plt.scatter(X[y_kmeans ==4, 0], X[y_kmeans == 4,1], s =100, c = "magenta", label ="Conservadores")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s =300, c = "yellow", label = "Baricentros")

plt.title("Cluster de Clientes")
plt.xlabel("Ingresos Anuales")
plt.ylabel("Puntuacion de Gastos")
plt.legend()
plt.show















