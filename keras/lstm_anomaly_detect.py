from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, Dropout
import numpy as np
from numpy import genfromtxt
from sklearn import preprocessing
from datetime import datetime

timesteps = 10

my_data = genfromtxt('../data/cf_07_03.csv', delimiter=',')
my_data = my_data[1:]
my_data_raw = np.nan_to_num(my_data)
my_data = np.delete(my_data_raw, 0, 1)
# min_max_scaler = preprocessing.MinMaxScaler()
# my_data = min_max_scaler.fit_transform(my_data)
my_data = preprocessing.scale(my_data)

'''
features: Process.RpcHandler.HttpQuery,Process.Incoming.RowkeyTemplate,Process.ConnectionManager.Open,Process.UniqueId.Got,Process.Query.RegexScan,Process.NioWorker.Read,Process.SaltScanner.MergeSort,Exception.MetricName.Invalid,Process.TSDB.AddPoint,Exception.Hbase.Remote.IO,Exception.Runtime.ShouldNotBeHere,Process.PutDataPointRpc,Success.Query.New,Exception.Stumbleupon.Defer,Warn.UniqueId.AlreadyWaiting.Assign,Process.ConnectionManager.HandleUpstream,Process.QueryRpc.TSQuery,Process.CompactionQueue.Flush,Process.Frame.Omit,Process.Put.Done,Query.Execute.New,Success.Stumbleupon.Defer,Exception.UniqueId.Assign,Query.RegionClient.Done,Exception.RegionClient.Decode,Exception.UniqueId.Fail.PendingAssignment,Exception.QueryRpc,Process.Remove.CompletedQuery,Process.UniqueId.Complete.PendingAssignment,Query.Completing,Process.SaltScanner.Scan.Complete"]
'''
# selected_feature_indexes = [1,2,7,8,9,11,12,15,16,18,19,20,23,26,29]
# my_data = my_data[:,selected_feature_indexes]
'''
Process.Rocksdb.L0.CommitSuccess,Process.Rocksdb.Compaction.MovedTo.L1,Exception.UpstreamServer.Connect,Exception.Cephx.CouldNotFindEntity,Process.Rocksdb.Compaction.Start2,Process.Rocksdb.Compaction.Interval,Report.Cluster.NewLeader,Process.Rocksdb.Compaction.L1,Process.Start.Request,Process.Rocksdb.Compaction.L2,Process.CheckSub.SendingToClient,Process.Scrub.Ok,Process.Rocksdb.GeneratTable,Process.Rocksdb.Compaction.OK,Report.Connection.Accept,Warn.Paxos.Recovering,Report.OsdPool.Enable,Process.Rocksdb.Compaction.Manual,Warn.UpstreamServer.TempDisabled,Process.Rocksdb.Compaction.Start,Process.Mon.Rank,Process.MDSStandby.Replay.Done,Process.Rocksdb.TableFileCreate,Process.Rocksdb.Delete.WAL,Process.Paxosservice.Upgraded,Process.Rocksdb.Compaction.MovingTo.L1,Warn.SlowRequest.Client,Process.Rocksdb.StopWrite.WaitFlush,Process.Rocksdb.Compaction.SyncingLog,Process.Rocksdb.Compaction.FlushMemToFile,Process.Rocksdb.L0.FlushStart,Report.Request.Done,Report.Server.ToStandby,Report.Rocksdb.Compaction.Finished,Process.Rocksdb.MemTable.Created
'''
# selected_feature_indexes = [0,2,4,5,7,9,10,13,21,25,28,29,30,33,34]
# my_data = my_data[:,selected_feature_indexes]


X_tmp = []
Y_tmp = []
X_tmp_raw = []
for i in range(len(my_data) - timesteps - 1):
    X_tmp.append(my_data[i:i + timesteps])
    Y_tmp.append(my_data[i + timesteps])
    X_tmp_raw.append(my_data_raw[i:i + timesteps])
X = np.array(X_tmp)
X_raw = np.array(X_tmp_raw)
Y = np.array(Y_tmp)

randomize = np.arange(len(X))
np.random.shuffle(randomize)
X = X[randomize]
X_raw = X_raw[randomize]
Y = Y[randomize]

input_dim = len(my_data[0])
output_dim = len(Y[0])

# 期望输入数据尺寸: (batch_size, timesteps, data_dim)
model = Sequential()
# model.add(LSTM(32, return_sequences=True, dropout=0.2, recurrent_dropout=0.2,
#                 input_shape=(timesteps, input_dim)))
model.add(LSTM(32, return_sequences=True,
                input_shape=(timesteps, input_dim)))
# model.add(LSTM(32, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
model.add(LSTM(16))
model.add(Dense(output_dim))

model.compile(loss='mse',
              # optimizer='adam')
              optimizer='rmsprop')
              # metrics=['accuracy'])

# 生成虚拟训练数据
val_length = len(X) // 3
train_length = len(X) - val_length
x_train = X[:train_length]
y_train = Y[:train_length]

# 生成虚拟验证数据
x_val = X[train_length:]
y_val = Y[train_length:]

model.fit(x_train, y_train,
          batch_size=32, epochs=200,
          validation_data=(x_val, y_val))

# dict = {}
y_preds = model.predict(X)
for i in range(len(X_raw)):
    loss = np.linalg.norm(y_preds[i] - Y[i])
    print(X_raw[i][timesteps - 1][0] + 60, datetime.fromtimestamp(X_raw[i][timesteps - 1][0] + 60), loss)
