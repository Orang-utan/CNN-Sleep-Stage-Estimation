import pickle
import numpy as np 
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn import preprocessing

X_raw = pickle.load(open("rr_record_total.p", "rb"))
Y_raw = pickle.load(open("rr_labels_one_hot_total.p", "rb"))\

X_raw = pickle.load(open("rr_record_total.p", "rb"))
Y_raw = pickle.load(open("rr_labels_one_hot_total.p", "rb"))

time_lag = 3 #3 minute time lag
max_rr_per_minute = 140
train_test_split = 0.8
output_class = 3

for i, patient_sleep_time in enumerate(X_raw):
    sleep_time_to_pad = len(patient_sleep_time) % time_lag

    if sleep_time_to_pad != 0:
        for _ in range(time_lag - sleep_time_to_pad):
            X_raw[i].append([0] * max_rr_per_minute)
            Y_raw[i].append([0] * output_class)

    if len(X_raw[i]) % time_lag != 0:
        print('Error: ', len(X_raw[i]))

    for j, rr_per_minute in enumerate(patient_sleep_time):
        init_rr_per_minute = len(rr_per_minute)

        #pad every patient's per minute RR record to 140 samples
        for _ in range(max_rr_per_minute - init_rr_per_minute):
            X_raw[i][j].append(0)

        #double check per minute RR record is padded to 140 samples
        if len(X_raw[i][j]) != max_rr_per_minute:
            print('Error: ', len(X_raw[i][j]))

X_flatten = []
Y_flatten = []
#flatten X_raw & Y_raw
for patient in X_raw:
    for sleep_time in patient:
        X_flatten.append(sleep_time)

for patient in Y_raw:
    for sleep_time in patient:
        Y_flatten.append(sleep_time)

X_processed = []
Y_processed = Y_flatten[time_lag - 1:]
for i in range(len(X_flatten)):
    if i < len(X_flatten) - (time_lag - 1):
        temp = []
        for j in range(time_lag):
            temp.append(X_flatten[i + j])
        X_processed.append(temp)

X_train_raw = X_processed[: int(len(X_processed)*train_test_split)] #27 samples
Y_train_raw = Y_processed[: int(len(Y_processed)*train_test_split)] #27 samples
X_test_raw = X_processed[int(len(X_processed)*train_test_split): ] #18 samples
Y_test_raw = Y_processed[int(len(Y_processed)*train_test_split): ] #18 samples

X_train = np.array(X_train_raw, dtype=np.float)
Y_train = np.array(Y_train_raw, dtype=np.float)
X_test = np.array(X_test_raw, dtype=np.float)
Y_test = np.array(Y_test_raw, dtype=np.float)
'''
max_val = np.amax(X_train)
X_train = X_train/max_val
X_test = X_test/max_val

mean_val = np.mean(X_train)
X_train = X_train - mean_val
X_test = X_test - mean_val
'''
model = load_model('time_lag_cnn_1.h5')
Y_predict = model.predict_classes(X_test)
np.set_printoptions(threshold=np.nan)
print(Y_predict)

print(len(Y_predict))
print(len(X_test))

time_step_to_predict = 1500

X_axis = []
Y_axis = Y_predict[1000:time_step_to_predict]
Y_axis = np.array(Y_axis)
Y_true = Y_test[1000:time_step_to_predict]
Y_true = np.array(Y_true)
Y_true = np.argmax(Y_true, axis=1)


time = list(range(time_step_to_predict - 1000))
for i in range(time_step_to_predict - 1000):
    X_axis.append(np.sum(X_test[i]))

X_1_np = np.reshape(X_axis, (-1, 1))
Y_1_np = np.reshape(Y_axis, (-1, 1))
Y_true_np = np.reshape(Y_true, (-1, 1))

min_max_scaler = preprocessing.MinMaxScaler()

X_1_np = min_max_scaler.fit_transform(X_1_np)
Y_1_np = min_max_scaler.fit_transform(Y_1_np)
Y_true_np = min_max_scaler.fit_transform(Y_true_np)

plt.plot(time, Y_1_np, marker='s')
plt.plot(time, Y_true_np, marker='o')
plt.show()



