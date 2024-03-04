"""
This script loads a trained linear regression model and makes predictions
on new data.

Dependencies:
- pandas: For reading CSV files and data manipulation.
- joblib: For loading the trained model from a file.
- sklearn.linear_model.LinearRegression: For making predictions using a
linear regression model.
- os: For setting the working directory.

Returns:
None
"""

import pandas as pd
import joblib
import yaml

# Abrir yaml
with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

# Leémos los datos de las casas a predecir
df = pd.read_csv(config['data']['clean']['test'])
X_pred = df.copy()
X_pred.drop(columns="Id", inplace=True)
X_pred = X_pred.to_numpy()

# Leémos el modelo
model = joblib.load('model.sav')

# Hacemos las predicciones para un nuevo conjunto de casas
precios_pred = model.predict(X_pred)

# Guardamos los resultados
df["SalePrice"] = precios_pred
df.to_csv(config['data']['predictions'], index=False)
