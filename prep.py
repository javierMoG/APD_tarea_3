"""
This script preprocesses data from two CSV files, one containing training
data and the other containing test data. It performs the following steps:

1. Reads the raw data from the CSV files and joins them to preprocess.
2. Drops columns with a high percentage of missing values.
3. Imputes missing values in the remaining columns, using the mean for numeric
variables and the mode for categorical variables.
4. Converts categorical variables into one-hot encoded format.
5. Prepares the preprocessed data for training and inference by splitting it
back into training and test sets.
6. Saves the preprocessed data to CSV files.

Dependencies:
- pandas: For reading CSV files and data manipulation.
- os: For setting the working directory.

Returns:
None
"""

import pandas as pd
import yaml
import logging
from datetime import datetime
from utils import load_data, write_data, fill_na, onehot

# Setup Logging
now = datetime.now()
date_time = now.strftime("%Y%m%d_%H%M%S")
log_prep_file_name = f"logs/{date_time}_prep.log"
logging.basicConfig(
    filename=log_prep_file_name,
    level=logging.DEBUG,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

# Abrir yaml
with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

# Leémos los datos sin procesar
logging.info('Cargando los datos crudos para su preprocesamiento')    
tbl_train = load_data(config['data']['raw']['train'])
tbl_test = load_data(config['data']['raw']['test'])
logging.debug(f'Número de datos de entrenamiento: {len(tbl_train)}')
logging.debug(f'Número de datos de prueba: {len(tbl_test)}')
logging.debug(f'Número de variables explicativas: {len(tbl_train.columns)}')


logging.info('Preprocesamiento de los datos')
# Guardamos la variable objetivo
y = tbl_train["SalePrice"]
# Eliminamos la columna de la variable objetivo de la tabla tbl_train
tbl_train.drop(columns=['SalePrice'], inplace=True)
# Unimos las tablas tbl_train y tbl_test para preprocesar los datos
df = pd.concat([tbl_train, tbl_test])
# Eliminamos las columnas con un alto porcentaje de valores nulos
df.drop(columns=['Alley', 'MasVnrType', 'FireplaceQu', 'PoolQC',
                 'Fence', 'MiscFeature'], inplace=True)
logging.debug(f'Número de variables que se utilizarán en el modelo: {len(df.columns)}')

# Imputamos los valores faltantes de las demás columnas
df = fill_na(df)

# Convertimos las variables categoricas a one hot encoding
df = onehot(df)

logging.debug(f"Número de columnas después de aplicar one-hot encoding a las variables categóricas {len(df.columns)}")
logging.info(f"Guardamos los nuevos datos preprocesados")
# Una vez que se procesaron lo datos preparamos las tablas tbl_train y tbl_test
tbl_train_clean = df[0:1460].copy()
tbl_test_clean = df[1460:2920].copy()
logging.debug(f'Número de datos de entrenamiento: {len(tbl_train_clean)}')
logging.debug(f'Número de datos de prueba: {len(tbl_test_clean)}')

# Agregamos a la tabla tbl_train_clean la variable objetivo
tbl_train_clean["SalePrice"] = y

# Guardamos los datos preprocesados
write_data(tbl_train_clean, config['data']['clean']['train'])
write_data(tbl_test_clean, config['data']['clean']['test'])
