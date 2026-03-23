import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

data = yf.download("AAPL", start="2020-01-01", end="2023-01-01")[['Close']]
values = data.values

train_size = int(len(values)*0.8)
train, test = values[:train_size], values[train_size:]


X = np.arange(len(values)).reshape(-1,1)
y = values

lr = LinearRegression()
lr.fit(X[:train_size], y[:train_size])
pred_lr = lr.predict(X[train_size:])


scaler = MinMaxScaler()
scaled = scaler.fit_transform(values)

def make_data(ds, step=10):
    X,Y=[],[]
    for i in range(len(ds)-step):
        X.append(ds[i:i+step])
        Y.append(ds[i+step])
    return np.array(X), np.array(Y)

train_s, test_s = scaled[:train_size], scaled[train_size:]
X_train,y_train = make_data(train_s)
X_test,y_test = make_data(test_s)

model = Sequential()
model.add(LSTM(32, input_shape=(10,1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=5, batch_size=16, verbose=0)

pred_lstm = model.predict(X_test)
pred_lstm = scaler.inverse_transform(pred_lstm)
y_test_real = scaler.inverse_transform(y_test)


print("Linear Regression MSE:", mean_squared_error(test, pred_lr))
print("LSTM MSE:", mean_squared_error(y_test_real, pred_lstm))


plt.plot(test, label="Actual")
plt.plot(pred_lr, label="Linear")
plt.plot(pred_lstm, label="LSTM")
plt.legend()
plt.title("Stock Prediction")
plt.show()

