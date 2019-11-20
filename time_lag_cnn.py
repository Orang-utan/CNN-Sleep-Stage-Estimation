import pickle
import numpy as np 
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv1D, BatchNormalization, GlobalAveragePooling1D
import autokeras as ak

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

X_train = np.divide(60000, X_train, where=X_train!=0)
X_test = np.divide(60000, X_test, where=X_test!=0)

max_val = X_train.max()

X_train *= np.divide(max_val, X_train, where=X_train!=0)
X_test *= np.divide(max_val, X_test, where=X_test!=0)

print(X_train.shape)
print(Y_train.shape)

print(X_test.shape)
print(Y_test.shape)

model = Sequential()

model.add(Conv1D(64, kernel_size=2, activation='relu', input_shape=(time_lag, max_rr_per_minute)))
model.add(BatchNormalization())
model.add(Conv1D(64, kernel_size=2, activation='relu'))
model.add(BatchNormalization())
#model.add(Flatten())
#model.add(Dropout(0.5))
model.add(GlobalAveragePooling1D())
model.add(Dense(output_class))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x=X_train, y=Y_train, epochs=100, validation_data=(X_test, Y_test))
model.save('time_lag_cnn_1.h5')



 



    

    

