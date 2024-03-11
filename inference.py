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
import logging
from datetime import datetime
from utils import load_data, write_data

# Setup Logging
now = datetime.now()
date_time = now.strftime("%Y%m%d_%H%M%S")
log_inference_file_name = f"logs/{date_time}_inference.log"
logging.basicConfig(
    filename=log_inference_file_name,
    level=logging.DEBUG,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

# Abrir yaml
with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

# Leémos los datos de las casas a predecir
logging.info("Leémos los datos limpios de las casas cuyo precio se quiere predecir")    
df = load_data(config['data']['clean']['test'])
X_pred = df.copy()
X_pred.drop(columns="Id", inplace=True)
X_pred = X_pred.to_numpy()

# Leémos el modelo
model = joblib.load('model.sav')

# Hacemos las predicciones para un nuevo conjunto de casas
precios_pred = model.predict(X_pred)

# Guardamos los resultados
logging.info("Guardamos las predicciones")
df["SalePrice"] = precios_pred
write_data(df, config['data']['predictions'])
