from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
from numpy import genfromtxt
from sklearn import preprocessing
from datetime import datetime

timesteps = 30
errorsteps = 5 # will have trouble in five minutes

def prepair_data(raw_features, statuses, timesteps):
    raw_features = raw_features[1:]
    train_data_raw = np.nan_to_num(raw_features)
    raw_features = np.delete(train_data_raw, 0, 1)
    min_max_scaler = preprocessing.MinMaxScaler()
    raw_features = min_max_scaler.fit_transform(raw_features)
    X_tmp = []
    Y_tmp = []
    X_tmp_raw = []

    for i in range(len(raw_features) - timesteps - 1):
        X_tmp.append(raw_features[i:i + timesteps])
        if statuses != None:
            status = 0
            for j in range(i + timesteps, min(i + timesteps + errorsteps, len(raw_features))):
                minute = train_data_raw[j][0]
                if minute in statuses and statuses[minute] == 2:
                    status = 1
                    break
            Y_tmp.append(status)
        X_tmp_raw.append(train_data_raw[i:i + timesteps])
    X_train = np.array(X_tmp)
    X_raw = np.array(X_tmp_raw)
    Y_train = np.array(Y_tmp)
    return (X_train, X_raw, Y_train)

def main():
    train_data = genfromtxt('../data/all_06_26.csv', delimiter=',')
    val_data = genfromtxt('../data/all_06_25.csv', delimiter=',')
    raw_statuses = genfromtxt('../data/status_06_26.csv', delimiter=',')

    minute_statuses = {}
    for status in raw_statuses:
        minute = status[0] // 60 * 60
        if minute not in minute_statuses or minute_statuses[minute] == 0:
            minute_statuses[minute] = status[1]

    X_train, X_raw, Y_train = prepair_data(train_data, minute_statuses, timesteps)
    X_val, X_val_raw, _ = prepair_data(val_data, None, timesteps)

    input_dim = X_train.shape[2]
    output_dim = 1 # binary classify

    # 期望输入数据尺寸: (batch_size, timesteps, data_dim)
    model = Sequential()
    model.add(LSTM(256, return_sequences=True,
                   input_shape=(timesteps, input_dim)))
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(256))
    model.add(Dense(output_dim, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model.fit(X_train, Y_train,
              batch_size=64, epochs=10,
              )
              # validation_data=(x_val, y_val))

    # dict = {}
    y_preds = model.predict(X_val)
    for i in range(len(X_val_raw)):
        print(datetime.fromtimestamp(X_val_raw[i][timesteps - 1][0]), round(y_preds[i][0], 2))

if __name__ == '__main__':
  main()
