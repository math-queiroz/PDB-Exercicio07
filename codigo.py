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

# Funcão para calcular o R2 Score ajustado
def r2_score_ajustado(x_test, x_pred):
    R2 = r2_score(x_test, x_pred)
    n = len(x)
    p = len(x[0])
    return 1-(1-R2)*(n-1)/(n-p-1)

# Reshape
y = y.reshape(-1, 1)

# Normalização dos dados
xScaler = StandardScaler()
yScaler = StandardScaler()
x = xScaler.fit_transform(x)
y = yScaler.fit_transform(y)

# Divisão dos dados em treino e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Títulos da tabela de saída
print(f'Modelo\t\t\t\tR2 Score')

# 1.Regressao linear
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
print(f'Regressão Linear:\t\t{r2_score_ajustado(y_test, y_pred):.5f}')

# 2.Regressao linear múltipla
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x_train)
regressor = LinearRegression()
regressor.fit(x_poly, y_train)
y_pred = regressor.predict(poly.transform(x_test))
print(f'Regressão Linear Múltipla:\t{r2_score_ajustado(y_test, y_pred):.5f}')

# 3.Regressao por vetor de suporte
regressor = SVR(kernel= 'rbf')
regressor.fit(x_train, y_train.ravel())
y_pred = regressor.predict(x_test)
print(f'Regressão Vetor de Suporte:\t{r2_score_ajustado(y_test, y_pred):.5f}')

# 4.Regressao por arvore de decisao
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
print(f'Regressão Arvore de Decisao:\t{r2_score_ajustado(y_test, y_pred):.5f}')

# 5.Regressao por floresta aleatoria
regressor = RandomForestRegressor(n_estimators=5, random_state=0)
regressor.fit(x_train, y_train.ravel())
y_pred = regressor.predict(x_test)
print(f'Regressão Floresta Aleatoria:\t{r2_score_ajustado(y_test, y_pred):.5f}')
