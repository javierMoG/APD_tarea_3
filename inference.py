import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
import os

# Dirección local del repo
os.chdir('c:\\Users\\javie\\OneDrive - INSTITUTO TECNOLOGICO AUTONOMO DE MEXICO\\MaestriaEnCienciaDeDatos\\4toSemestre\\ArquitecturaDeProductosDeDatos\\Tareas\\Tarea3\\APD_tarea_3')

# Leémos los datos de las casas a predecir
df = pd.read_csv("./data/inference.csv")
X_pred = df.copy()
X_pred.drop(columns="Id", inplace=True)
X_pred = X_pred.to_numpy()

# Leémos el modelo
model = joblib.load('model.sav')

# Hacemos las predicciones para un nuevo conjunto de casas
precios_pred = model.predict(X_pred)

# Guardamos los resultados
df["SalePrice"] = precios_pred
df.to_csv("./data/predictions.csv", index = False)