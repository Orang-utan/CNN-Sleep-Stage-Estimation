import pickle
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import preprocessing
import scipy.optimize as opt

X_data = pickle.load(open("rr_record_total.p", "rb"))
Y_data = pickle.load(open("rr_labels_total.p", "rb"))

X_1_raw = X_data[2]
X_1 = []
Y_1 = Y_data[2]

time = list(range(len(Y_1)))

for minute_record in X_1_raw:
    sum = 0.0
    for i in range(len(minute_record)):
        sum += int(minute_record[i])
    mean = sum/float(len(minute_record))
    MHR = 60000/mean
    X_1.append(MHR)

X_1_np = np.array(X_1)
Y_1_np = np.array(Y_1)
Y_1_np = Y_1_np.astype(np.float64)

X_1_np = np.reshape(X_1_np, (-1, 1))
Y_1_np = np.reshape(Y_1_np, (-1, 1))

min_max_scaler = preprocessing.MinMaxScaler()

X_1_np = min_max_scaler.fit_transform(X_1_np)
Y_1_np = min_max_scaler.fit_transform(Y_1_np)


plt.plot(time, X_1_np, marker='s')
plt.plot(time, Y_1_np, marker='o')
plt.show()






    