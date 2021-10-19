# pvsd
Predicción de valores de un stock determinado usando Deep Learning
# Prediccion de valores de un stock determinado

![alt text](https://github.com/Jaerab/pvsd/blob/main/StockPrediction.png?raw=true)

# Resumen

Este proyecto tuvo como objetivo principal la predicción del valor de las acciones de una compañía específica, principalmente se hizo para observar qué tan precisos serán las predicciones de los algoritmos de Inteligencia Artificial, para ello se hizo una comparación entre múltiples algoritmos de regresión y luego algunos algoritmos RNN con el fin de encontrar los mejores resultados.

# Instalacion

Primero debemos instalar en el notebook yfinance

```sh
!pip install yfinance --upgrade --no-cache-dir
```

# Librerias

Para el proyecto es necesario importar las siguientes librerias

```sh
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from pandas_datareader import data as pdr
import yfinance as yf
from sklearn.model_selection import train_test_split
```
