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

# Leémos los datos sin procesar
tbl_train = pd.read_csv("./data/train.csv")
tbl_test = pd.read_csv("./data/test.csv")

# Guardamos la variable objetivo
y = tbl_train["SalePrice"]
# Eliminamos la columna de la variable objetivo de la tabla tbl_train
tbl_train.drop(columns=['SalePrice'],inplace=True)
# Unimos las tablas tbl_train y tbl_test para preprocesar los datos
df = pd.concat([tbl_train, tbl_test])
# Eliminamos las columnas con un alto porcentaje de valores nulos
df.drop(columns=['Alley', 'MasVnrType', 'FireplaceQu', 'PoolQC', \
                 'Fence', 'MiscFeature'],inplace=True)
# Imputamos los valores faltantes de las demás columnas
def fill_na(tbl):
    """
    Fills missing values in a DataFrame.

    Iterates over each column in the DataFrame and fills missing values using
    appropriate strategies:
    - For numeric columns, missing values are filled with the mean of the column.
    - For categorical columns, missing values are filled with the mode of the column.

    Parameters:
    - tbl (DataFrame): The input DataFrame containing missing values to be filled.

    Returns:
    DataFrame: DataFrame with missing values filled.
    """
    for column in tbl.columns:
        if tbl[column].isnull().any():
            if pd.api.types.is_numeric_dtype(tbl[column]):
                #Imputamos con la media para la variables numéricas
                tbl[column].fillna(tbl[column].mean(), inplace=True)
            else:
                #Imputamos con la moda para las variables categóricas
                tbl[column].fillna(str(tbl[column].mode()), inplace=True)
    return tbl

df = fill_na(df)

# Convertimos las variables categoricas a one hot encoding
def onehot(tbl):
    """
    Converts categorical variables in a DataFrame into one-hot encoded format.

    Iterates over each column in the DataFrame and converts non-numeric columns
    into one-hot encoded format.
    
    Parameters:
    - tbl (DataFrame): The input DataFrame containing categorical variables to be one-hot encoded.

    Returns:
    DataFrame: DataFrame with categorical variables converted into one-hot encoded format.
    """
    df_aux=tbl
    i=0
    for column in tbl.columns:
        if not pd.api.types.is_numeric_dtype(tbl[column]):
            # print(column)
            df1=pd.get_dummies(tbl[column],drop_first=True)
            tbl.drop([column],axis=1,inplace=True)
            if i==0:
                df_aux=df1.copy()
            else:
                df_aux=pd.concat([df_aux,df1],axis=1)
            i=i+1
    df_aux=pd.concat([tbl,df_aux],axis=1)
    return df_aux
df = onehot(df)

# Una vez que se procesaron lo datos preparamos las tablas tbl_train y tbl_test
tbl_train_clean = df[0:1461].copy()
tbl_test_clean = df[1461:2920].copy()

# Agregamos a la tabla tbl_train_clean la variable objetivo
tbl_train_clean["SalePrice"] = y

# Guardamos los datos preprocesados
tbl_train_clean.to_csv("./data/prep.csv", index = False)
tbl_test_clean.to_csv("./data/inference.csv", index = False)
