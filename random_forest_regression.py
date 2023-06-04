# Regresio con Bosques Aleatorios

#Carga de Librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Carga de Dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Modelo de Regresion con Bosques Aleatorios
# n_estimators es la cantidad de arboles que vamos a utilizar
#El Valor esperado es 160,000 en el sueldo de un empleado
#Con el nivel 6.5
from sklearn.ensemble import RandomForestRegressor
regression = RandomForestRegressor( n_estimators = 10, random_state = 0)
regression.fit(X, y)

y_pred = regression.predict([[6.5]])

#Visualizacion 
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X,y, color ="red")
plt.plot(X_grid, regression.predict(X_grid), color ="blue")
plt.title("Modelo de Regresion con Bosques Aleatorios")
plt.xlabel("Posicion del empleado")
plt.ylabel("Salario")