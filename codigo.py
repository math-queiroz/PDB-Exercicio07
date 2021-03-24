# Imports principais
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Imports do sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Import do dataset
dataset = pd.read_csv('winequality-red.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Array de algoritmos
algoritmos = ['Regressão Linear', 'Regressão Linear Múltipla', 'Regressão Vetor de Suporte', 'Regressão Arvore de Decisao', 'Regressão Floresta Aleatoria']

# Funcão para calcular o R2 Score ajustado
def r2_score_ajustado(x_test, x_pred):
    R2 = r2_score(x_test, x_pred)
    n = len(x)
    p = len(x[0])
    return 1-(1-R2)*(n-1)/(n-p-1)

# Reshape para poder calcular a regressao por vetor de suporte
y = y.reshape(-1, 1)

# Normalização dos dados
xScaler = StandardScaler()
yScaler = StandardScaler()
x = xScaler.fit_transform(x)
y = yScaler.fit_transform(y)

# Divisão dos dados em treino e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Títulos da tabela de saída
print('Modelo\t\t\t\tR2 Score')

# 1.Regressao linear
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
R2 = r2_score_ajustado(y_test, y_pred)
print(f'{algoritmos[0]}:\t\t{R2:.5f}')
melhor_R2 = (0, R2)

# 2.Regressao linear múltipla
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x_train)
regressor = LinearRegression()
regressor.fit(x_poly, y_train)
y_pred = regressor.predict(poly.transform(x_test))
R2 = r2_score_ajustado(y_test, y_pred)
print(f'{algoritmos[1]}:\t{R2:.5f}')
if (R2 > melhor_R2[1]):
    melhor_R2 = (1, R2)

# 3.Regressao por vetor de suporte
regressor = SVR(kernel= 'rbf')
regressor.fit(x_train, y_train.ravel())
y_pred = regressor.predict(x_test)
R2 = r2_score_ajustado(y_test, y_pred)
print(f'{algoritmos[2]}:\t{R2:.5f}')
if (R2 > melhor_R2[1]):
    melhor_R2 = (2, R2)

# 4.Regressao por arvore de decisao
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
R2 = r2_score_ajustado(y_test, y_pred)
print(f'{algoritmos[3]}:\t{R2:.5f}')
if (R2 > melhor_R2[1]):
    melhor_R2 = (3, R2)

# 5.Regressao por floresta aleatoria
regressor = RandomForestRegressor(n_estimators=5, random_state=0)
regressor.fit(x_train, y_train.ravel())
y_pred = regressor.predict(x_test)
R2 = r2_score_ajustado(y_test, y_pred)
print(f'{algoritmos[4]}:\t{R2:.5f}')
if (R2 > melhor_R2[1]):
    melhor_R2 = (4, R2)

# Exibição do resultado do algoritmo com melhor resultado de pontuação por R²
print(f'\nO melhor algoritmo é {algoritmos[melhor_R2[0]]} com uma pontuação por R² de {melhor_R2[1]}!')