#don't forget upload the csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

gdp = pd.read_csv("gdp_data.csv", header=2)
    
gdp.info()

# Select rows where the "Country Name" column contains "World"
world = gdp.loc[gdp["Country Name"].str.contains("World")]

#borramos primera fila mundo arabe
world = world.drop(world.index[0])


# Display the rows of the DataFrame with their indices
world.reset_index(inplace=True)


world = world.rename(columns={"Country Name": "Index"})

# Remove the columns 0, 2, 3, and 4 from the DataFrame
world.drop(["index", "Country Code", "Indicator Name", "Indicator Code"], axis=1, inplace=True)

# Display the modified DataFrame
world.head()

# transpose
wt = world.transpose()

# Eliminar el encabezado del dataframe
header = wt.iloc[0]
df = wt[1:]
# Asignar la primera fila como encabezado
df.columns = header

df.reset_index(inplace=True)

df.index

df = df.rename(columns={"index": "Year", "World": "GDP (current US$)"})

#revisamos que los datos esten listos para correr la regresion
df.info()

#reemplazamos espacios y comas
df['Year'] = df['Year'].str.replace(',', '')
df['GDP (current US$)'] = df['GDP (current US$)'].str.replace(',', '')
df['Year'] = df['Year'].str.replace(' ', '')
df['GDP (current US$)'] = df['GDP (current US$)'].str.replace(' ', '')

df

#tenemos datos tipo string y los vamos a pasar a valores enteros

# selecciona las columnas "Year" y "GDP" del dataframe original
df_temp = df[["Year", "GDP (current US$)"]]

# convierte las columnas seleccionadas a tipo "int64"
df_temp = df_temp.apply(pd.to_numeric, downcast='integer')

# sobreescribe las columnas originales con los datos convertidos
df[["Year", "GDP (current US$)"]] = df_temp

#dos visualizaciones a ver que pedo con los datos
plt.plot(df["Year"], df["GDP (current US$)"])
plt.xlabel("Year")
plt.ylabel("GDP (current US$)")
plt.title("GDP 1960-2021")
plt.show()

plt.scatter(df["Year"], df["GDP (current US$)"])
plt.xlabel("Year")
plt.ylabel("GDP (current US$)")
plt.title("GDP 1960-2021")
plt.show()

#creacion de variables predictoras y de respuesta, predictores con los valores de la columna a??o, y de respuesta con los valores de la columna gdp
predictors = df[["Year"]]
response = df[["GDP (current US$)"]]

# Divide los datos en conjuntos de entrenamiento y evaluaci??n
X_train, X_test, y_train, y_test = train_test_split(predictors, response, test_size=0.2)

# Create a linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
predictions = model.predict(X_test)

print(type(predictions))

# Grafica los puntos de entrenamiento
plt.scatter(X_train, y_train, label="Entrenamiento", color="red")

#Grafica los puntos de las predicciones
plt.scatter(X_test, predictions, label="Predicciones", color='blue')

# Establece el t??tulo y las etiquetas de los ejes
plt.title("Regresi??n lineal")
plt.xlabel("A??os")
plt.ylabel("PIB")

# Muestra la leyenda
plt.legend()

# Muestra la gr??fica
plt.show()

#coeficiente de determinaci??n
r2 = model.score(X_test, y_test)
print(f"Coeficiente de determinaci??n: {r2}")

#error absoluto medio
mae = mean_absolute_error(y_test, predictions)
print(f'El error absoluto medio es: {mae}')

#error cuadr??tico medio
mse = mean_squared_error(y_test, predictions)
print(f"Error cuadr??tico medio: {mse}")

#raiz error cuadratico medio
rmse = np.sqrt(mse)
print(f'La raiz del error cuadratico medio es: {rmse}')
