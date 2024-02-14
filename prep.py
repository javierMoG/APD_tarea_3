import pandas as pd
import os

os.chdir('c:\\Users\\javie\\OneDrive - INSTITUTO TECNOLOGICO AUTONOMO DE MEXICO\\MaestriaEnCienciaDeDatos\\4toSemestre\\ArquitecturaDeProductosDeDatos\\Tareas\\Tarea3\\APD_tarea_3')

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
df.drop(columns=['Alley', 'MasVnrType', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'],inplace=True)
# Imputamos los valores faltantes de las demás columnas
def fill_na(df):
    for column in df.columns:
        if df[column].isnull().any():
            if pd.api.types.is_numeric_dtype(df[column]):
                #Imputamos con la media para la variables numéricas
                df[column].fillna(df[column].mean(), inplace=True)
            else:
                #Imputamos con la moda para las variables categóricas
                df[column].fillna(str(df[column].mode()), inplace=True)
    return df

df = fill_na(df)

# Convertimos las variables categoricas a one hot encoding
def onehot(df):
    df_aux=df
    i=0 
    for column in df.columns:
        if not(pd.api.types.is_numeric_dtype(df[column])):
            # print(column)
            df1=pd.get_dummies(df[column],drop_first=True)
            
            df.drop([column],axis=1,inplace=True)
            if i==0:
                df_aux=df1.copy()
            else:  
                df_aux=pd.concat([df_aux,df1],axis=1)
            i=i+1
    df_aux=pd.concat([df,df_aux],axis=1)        
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