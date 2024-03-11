"""
This script trains a linear regression model on preprocessed data and
saves the trained model.

Dependencies:
- pandas: For reading CSV files and data manipulation.
- os: For setting the working directory.
- sklearn.model_selection.train_test_split: For splitting the data into
training and test sets.
- sklearn.linear_model.LinearRegression: For training a linear regression
model.
- joblib: For saving the trained model to a file.

Returns:
None
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import yaml
import random
import logging
from datetime import datetime
from utils import load_data, write_data

# Setup Logging
now = datetime.now()
date_time = now.strftime("%Y%m%d_%H%M%S")
log_train_file_name = f"logs/{date_time}_train.log"
logging.basicConfig(
    filename=log_train_file_name,
    level=logging.DEBUG,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

# Abrir yaml
with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

# Leémos los datos procesados
logging.info(
    "Cargamos los datos de entrenamiento limpios para entrenar el modelo"
    )
df = load_data(config['data']['clean']['train'])

# Eliminamos la columna de Id para entrenar el modelo
df.drop(columns="Id", inplace=True)
X = df.loc[:, df.columns != 'SalePrice'].to_numpy()
Y = df["SalePrice"].to_numpy()

# Fijamos la semilla
logging.info(f"Semilla del modelo: {config['modeling']['random_seed']}")
random.seed(config['modeling']['random_seed'])

# Separamos los datos en conjunto de entrenamiento y de prueba
logging.info("Separamos los datos en conjunto de entrenamiento y prueba")
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

logging.debug(f"{len(x_train)} registros en el conjunto de entrenamiento")
logging.debug(f"{len(x_test)} registros en el conjunto de prueba")

# Entrenamos el modelo
logging.info("Entrenamos el modelo y lo evaluamos")
linreg = LinearRegression()
linreg.fit(x_train, y_train)

# Registramos el rendimiento del modelo
logging.debug(
    f"R^2 en el conjunto de entrenamiento:{linreg.score(x_train, y_train):.3f}"
    )
logging.debug(f"R^2 en el conjunto de prueba:{linreg.score(x_test, y_test):.3f}")

# Guardamos el modelo
logging.info("Guardamos el modelo")
joblib.dump(linreg, 'model.sav')
