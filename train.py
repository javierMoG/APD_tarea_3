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

# Leémos los datos procesados
df = pd.read_csv("./data/prep.csv")

# Eliminamos la columna de Id para entrenar el modelo
df.drop(columns="Id", inplace=True)
X = df.loc[:, df.columns != 'SalePrice'].to_numpy()
Y = df["SalePrice"].to_numpy()

# Separamos lo datos en conjunto de entrenamiento y de prueba (80% y 20%
# respectivamente)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

print(f"{len(x_train)} registros en el conjunto de entrenamiento")
print(f"{len(x_test)} registros en el conjunto de prueba")

# Entrenamos el modelo
linreg = LinearRegression()
linreg.fit(x_train, y_train)

# Imprimimos el rendimiento del modelo
print(
    f"R^2 en el conjunto de entrenamiento:{linreg.score(x_train, y_train):.3f}"
    )
print(f"R^2 en el conjunto de prueba:{linreg.score(x_test, y_test):.3f}")

# Guardamos el modelo
joblib.dump(linreg, 'model.sav')
