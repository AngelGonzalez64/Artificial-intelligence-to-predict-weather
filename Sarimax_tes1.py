import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tkinter import *
import concurrent.futures

# Leer los datos
df = pd.read_csv('Entrenamiento.csv')

# Convertir la columna de fecha a un formato de fecha de Python
df['fecha'] = pd.to_datetime(df['fecha'], format='%d/%m/%Y')

# Ordenar por fecha
df = df.sort_values('fecha')

# Normalizar las características para mejorar el rendimiento de LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df.iloc[:, 1:])

# Dividir los datos en entrenamiento y prueba
train_size = int(len(df_scaled) * 0.8)
train, test = df_scaled[0:train_size, :], df_scaled[train_size:len(df_scaled), :]

# Convertir un arreglo de valores en una matriz de conjuntos de datos
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, :])
    return np.array(dataX), np.array(dataY)

# Usar una ventana de 3 días para predecir el clima
look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# Cambiar la forma de [muestras, características] a [muestras, pasos de tiempo, características]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], trainX.shape[2]))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], testX.shape[2]))

# Crear y ajustar el modelo LSTM bidireccional
model = Sequential()
model.add(Bidirectional(LSTM(8, input_shape=(look_back, trainX.shape[2]))))
model.add(Dense(trainY.shape[1]))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=16, verbose=2)

# Hacer predicciones
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# Invertir las predicciones a escala original
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform(trainY)
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform(testY)

# Calcular el error cuadrático medio
trainScore = np.sqrt(mean_squared_error(trainY, trainPredict))
print('Resultado del entrenamiento: %.2f RMSE' % trainScore)
testScore = np.sqrt(mean_squared_error(testY, testPredict))
print('Resultado de la prueba: %.2f RMSE' % testScore)

# Definir la función que se ejecutará cuando se haga clic en el botón "Predecir"
def predecir():
    # Declarar df y df_scaled como global para poder usarlas dentro de la función
    global df, df_scaled

    # Obtener el número de días de la caja de entrada
    days_to_predict = int(entry.get())

    # Predecir los siguientes días
    future_predictions = []
    for i in range(days_to_predict):
        # Obtener el último conjunto de datos de entrada de entrenamiento
        last_data = df_scaled[-look_back:, :]

        # Reshape y hacer la predicción
        last_data = np.reshape(last_data, (1, look_back, df_scaled.shape[1]))
        prediction = model.predict(last_data)

        # Invertir la predicción a escala original
        prediction = scaler.inverse_transform(prediction)

        # Agregar la predicción a df y df_scaled
        next_date = df['fecha'].iloc[-1] + pd.DateOffset(days=1)
        df.loc[len(df)] = [next_date] + prediction.flatten().tolist()
        df_scaled = scaler.transform(df.iloc[:, 1:])

        future_predictions.append(prediction.flatten().tolist())

    # Mostrar las predicciones en la ventana de texto
    output_text.insert(END, "Fecha\t\tTemperatura Media\tPrecipitación Total\tVelocidad del Viento\n")
    for i in range(len(future_predictions)):
        date = df['fecha'].iloc[train_size + look_back + i]
        temp = future_predictions[i][0]
        precip = future_predictions[i][1]
        wind = future_predictions[i][2]
        output_text.insert(END, f"{date}\t{temp}\t\t\t{precip}\t\t\t\t{wind}\n")

    # Graficar los resultados
    plot_results()

# Crear una nueva ventana
window = Tk()

# Configurar la ventana
window.title("Predicción del Clima")
window.geometry("800x600")

# Crear una etiqueta y una entrada para ingresar el número de días a predecir
label = Label(window, text="Ingrese el número de días a predecir:")
label.pack()
entry = Entry(window)
entry.pack()

# Crear un botón para realizar la predicción
button = Button(window, text="Predecir", command=predecir)
button.pack()

# Crear un widget de texto para mostrar las predicciones
output_text = Text(window, height=10, width=60)
output_text.pack()

# Crear una figura y un objeto de eje para la gráfica
fig, ax = plt.subplots()

# Función para graficar los resultados
def plot_results():
    # Graficar los datos de entrenamiento, prueba y predicción
    dates_train = df['fecha'].iloc[:train_size]
    dates_test = df['fecha'].iloc[train_size + look_back:]
    dates_pred = df['fecha'].iloc[train_size + look_back:]

    temp_train = df['temperatura_media'].iloc[:train_size]
    temp_test = df['temperatura_media'].iloc[train_size + look_back:]
    temp_pred = df['temperatura_media'].iloc[train_size + look_back:]

    precip_train = df['precipitacion_total'].iloc[:train_size]
    precip_test = df['precipitacion_total'].iloc[train_size + look_back:]
    precip_pred = df['precipitacion_total'].iloc[train_size + look_back:]

    wind_train = df['velocidad_viento'].iloc[:train_size]
    wind_test = df['velocidad_viento'].iloc[train_size + look_back:]
    wind_pred = df['velocidad_viento'].iloc[train_size + look_back:]

    ax.clear()
    ax.plot(dates_train, temp_train, label='Entrenamiento (Temperatura Media)', color='blue')
    ax.plot(dates_test, temp_test, label='Prueba (Temperatura Media)', color='orange')
    ax.plot(dates_pred, temp_pred, label='Predicción (Temperatura Media)', color='green')
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.legend()
    ax.set_title('Predicción de la Temperatura Media')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Temperatura Media')

    # Mostrar la gráfica en la ventana
    plt.tight_layout()
    plt.show()

# Ejecutar la ventana principal
window.mainloop()
