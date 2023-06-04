#ANN

# ----- PREPROCESADO DE DATOS -----

# Cargar Librerias
import numpy as np
import matplotlib as plt
import pandas as pd

# Cargar de Dataset
dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:,13].values

#Variables Dummys
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Tranformacion para "Country"
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
#Trnasformacion para "Gender"
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

# Dvidir Entrenamiento y Testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

#Escalado de Variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# ----- CONSTRUIR ANN -----
# Importar Librerias para ANN
import keras
from keras.models import Sequential
from keras.layers import Dense

# Inicializar la ANN
classifier = Sequential()

# Capa de Entrada y Primera Capa Oculta 
classifier.add(Dense(units = 6, kernel_initializer = "uniform",
                     activation = "relu", input_dim = 11))

# Segunda Capa Oculta
classifier.add(Dense(units = 6, kernel_initializer = "uniform",
                     activation = "relu"))

# Capa de Salida
classifier.add(Dense(units = 1, kernel_initializer = "uniform",
                     activation = "sigmoid"))

#Compilar ANN
classifier.compile(optimizer = "adam", loss = "binary_crossentropy",
                   metrics = ["accuracy"])

# Ajustar ANN al Conjunto de Entrenamiento
classifier.fit(X_train, y_train, batch_size =10, epochs = 100)


# ----- EVALUACION DE MODELO Y CALCULO DE PREDICCIONES -----

#Prediccion de Resultados del Conjunto Testing
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

#Matriz de Confusion
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)


