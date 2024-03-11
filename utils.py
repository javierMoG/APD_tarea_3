import pandas as pd
import logging

def fill_na(tbl):
    """
    Fills missing values in a DataFrame.

    Iterates over each column in the DataFrame and fills missing values using
    appropriate strategies:
    - For numeric columns, missing values are filled with the mean of the
    column.
    - For categorical columns, missing values are filled with the mode of
    the column.

    Parameters:
    - tbl (DataFrame): The input DataFrame containing missing values to be
    filled.

    Returns:
    DataFrame: DataFrame with missing values filled.
    """
    for column in tbl.columns:
        if tbl[column].isnull().any():
            if pd.api.types.is_numeric_dtype(tbl[column]):
                # Imputamos con la media para la variables numéricas
                tbl[column].fillna(tbl[column].mean(), inplace=True)
            else:
                # Imputamos con la moda para las variables categóricas
                tbl[column].fillna(str(tbl[column].mode()), inplace=True)
    return tbl


def onehot(tbl):
    """
    Converts categorical variables in a DataFrame into one-hot encoded format.

    Iterates over each column in the DataFrame and converts non-numeric columns
    into one-hot encoded format.

    Parameters:
    - tbl (DataFrame): The input DataFrame containing categorical variables to
    be one-hot encoded.

    Returns:
    DataFrame: DataFrame with categorical variables converted into one-hot
    encoded format.
    """
    df_aux = tbl
    i = 0
    for column in tbl.columns:
        if not pd.api.types.is_numeric_dtype(tbl[column]):
            # print(column)
            df1 = pd.get_dummies(tbl[column], drop_first=True)
            tbl.drop([column], axis=1, inplace=True)
            if i == 0:
                df_aux = df1.copy()
            else:
                df_aux = pd.concat([df_aux, df1], axis=1)
            i = i+1
    df_aux = pd.concat([tbl, df_aux], axis=1)
    return df_aux


def empty_df(df, file_path):
    """
    Check if DataFrame is empty.

    This function checks if the given DataFrame is empty and logs a message indicating 
    whether the DataFrame is empty or not.

    :param df: DataFrame to be checked.
    :type df: pandas.DataFrame
    :param file_path: Path of the file associated with the DataFrame.
    :type file_path: str
    """
    try:
        assert len(df) != 0
        logging.info(f"El archivo {file_path} no está vacío")
    except AssertionError:
        logging.error(f"No hay datos en el archivo {file_path}")


def load_data(file_path):
    """
    Load data from a CSV file.

    This function reads data from a CSV file located at the specified file path 
    and returns a DataFrame containing the data.

    :param file_path: Path to the CSV file.
    :type file_path: str
    :return: DataFrame containing the data from the CSV file.
    :rtype: pandas.DataFrame
    """
    try:
        df = pd.read_csv(file_path)
        empty_df(df, file_path)
        return df
    except FileNotFoundError:
        logging.error(f"El archivo no está en este path: {file_path}")
        logging.error(f"Busca en otro lugar, por lo pronto no podemos proceder.")


def write_data(file, file_path):
    """
    Write DataFrame to a CSV file.

    This function writes the given DataFrame to a CSV file specified by the file path.

    :param file: DataFrame to be written to the CSV file.
    :type file: pandas.DataFrame
    :param file_path: Path to save the CSV file.
    :type file_path: str
    """
    try:
        file.to_csv(file_path, index=False)
        logging.debug(f'Archivo {file_path} guardado con éxito')
    except FileNotFoundError:
        logging.error(f"No existe la ruta: {file_path}")


