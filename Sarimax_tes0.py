import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM
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
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# Usar una ventana de 3 días para predecir el clima
look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# Cambiar la forma de [muestras, características] a [muestras, pasos de tiempo, características]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], trainX.shape[2]))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], testX.shape[2]))

# Crear y ajustar el modelo LSTM
model = Sequential()
model.add(LSTM(4, input_shape=(look_back, trainX.shape[2])))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# Hacer predicciones
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# Invertir las predicciones a escala original
trainPredict = scaler.inverse_transform(np.hstack((trainPredict, np.zeros((trainPredict.shape[0], df_scaled.shape[1] - 1)))))
trainPredict = trainPredict[:, 0]
testPredict = scaler.inverse_transform(np.hstack((testPredict, np.zeros((testPredict.shape[0], df_scaled.shape[1] - 1)))))
testPredict = testPredict[:, 0]

# Calcular el error cuadrático medio
trainScore = np.sqrt(mean_squared_error(trainY, trainPredict))
print('Resultado del entrenamiento: %.2f RMSE' % (trainScore))
testScore = np.sqrt(mean_squared_error(testY, testPredict))
print('Resultado de la prueba: %.2f RMSE' % (testScore))

# Definir la función que se ejecutará cuando se haga clic en el botón "Predecir"
def predecir():
    # Declarar df y df_scaled como global para poder usarlas dentro de la función
    global df, df_scaled

    # Obtener el número de días de la caja de entrada
    days_to_predict = int(entry.get())

    # Preparar los datos para las predicciones futuras
    predictions = pd.date_range(start=df['fecha'].iloc[-1] + pd.DateOffset(1), periods=days_to_predict)

    # Predecir el clima para el futuro utilizando hilos
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_predictions = [executor.submit(predict_next_day, df_scaled, predictions[i]) for i in range(days_to_predict)]

        # Obtener los resultados de las predicciones
        next_pred_unscaled = []
        for future in concurrent.futures.as_completed(future_predictions):
            next_pred_unscaled.append(future.result())

    # Actualizar los datos y visualizar las predicciones
    for i in range(days_to_predict):
        #next_row_scaled = np.hstack((next_pred_unscaled[i], np.zeros((next_pred_unscaled[i].shape[0], df_scaled.shape[1] - 1))))
        next_row_scaled = np.hstack((next_pred_unscaled[i].reshape(-1, 1), np.zeros((next_pred_unscaled[i].shape[0], df_scaled.shape[1] - 1))))

        df_scaled = np.vstack((df_scaled, next_row_scaled))
        df = df.append(pd.DataFrame({'fecha': predictions[i], 'temperatura_media': next_pred_unscaled[i][0]}, index=[0]), ignore_index=True)

    # Visualizar las predicciones
    plt.figure(figsize=(8, 6))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.plot(df['fecha'][:len(train) + look_back], df['temperatura_media'][:len(train) + look_back], label='Entrenamiento')
    plt.plot(df['fecha'][len(train) + look_back:len(df) - days_to_predict], df['temperatura_media'][len(train) + look_back:len(df) - days_to_predict], label='Prueba')
    plt.plot(df['fecha'][len(df) - days_to_predict:], df['temperatura_media'][len(df) - days_to_predict:], label='Predicción')
    plt.title('Predicción del clima para el próximo año')
    plt.xlabel('Fecha')
    plt.ylabel('Temperatura Media')
    plt.legend()
    plt.show()

# Función para predecir un solo día utilizando el modelo LSTM
def predict_next_day(data, date):
    last_features = data[-look_back:]
    last_features = last_features.reshape((1, look_back, df_scaled.shape[1]))
    next_pred = model.predict(last_features)
    next_pred_unscaled = scaler.inverse_transform(np.hstack((next_pred, np.zeros((next_pred.shape[0], df_scaled.shape[1] - 1)))))
    next_pred_unscaled = next_pred_unscaled[0, :]
    return next_pred_unscaled

# Crear la GUI
root = Tk()

# Crear una etiqueta y una caja de entrada para el número de días
label = Label(root, text="Ingrese el número de días para las predicciones después de 2022:")
entry = Entry(root)

# Crear un botón que llame a la función "predecir" cuando se haga clic en él
button = Button(root, text="Predecir", command=predecir)

# Colocar los widgets en la ventana
label.pack()
entry.pack()
button.pack()

# Ejecutar la GUI
root.mainloop()
