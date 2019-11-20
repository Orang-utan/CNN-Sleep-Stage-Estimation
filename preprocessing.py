import pickle
import numpy as np 
#from keras import preprocessing

X_raw = pickle.load(open("rr_record_total.p", "rb"))
Y_raw = pickle.load(open("rr_labels_one_hot_total.p", "rb"))

#Preprocessing Step 1:
#X_raw data dimensions (N * T * D); 
#N = # of Patients; T = Sleep Time (Min); D = RR Record per Minute
#T and D both varies depending on the patient
#Thus, below pad data with zeros in order to make dimensions the same.
#Expected Dimensions: (45 * 720 * 140);

max_sleep_minute = 720
max_rr_per_minute = 140
train_test_split = 0.6

for i, patient_sleep_time in enumerate(X_raw):
    init_sleep_time = len(patient_sleep_time)
    
    #pad every patient's sleep time to 720 minute
    for _ in range(max_sleep_minute - init_sleep_time):
        X_raw[i].append([0] * max_rr_per_minute)
        #pad y data as well
        Y_raw[i].append([0] * 4)

    #double check sleep time is padded to 720 minutes
    if len(X_raw[i]) != max_sleep_minute:
        print('Error: ', len(X_raw[i]))

    for j, rr_per_minute in enumerate(patient_sleep_time):
        init_rr_per_minute = len(rr_per_minute)

        #pad every patient's per minute RR record to 140 samples
        for _ in range(max_rr_per_minute - init_rr_per_minute):
            X_raw[i][j].append(0)

        #double check per minute RR record is padded to 140 samples
        if len(X_raw[i][j]) != max_rr_per_minute:
            print('Error: ', len(X_raw[i][j]))

#Preprocessing Step 2:
#Divide initial X/Y raw into test/train
#Convert X/Y train/test into numpy array
X_train_raw = X_raw[: int(len(X_raw)*train_test_split)] #27 samples
Y_train_raw = Y_raw[: int(len(Y_raw)*train_test_split)] #27 samples
X_test_raw = X_raw[int(len(X_raw)*train_test_split): ] #18 samples
Y_test_raw = Y_raw[int(len(Y_raw)*train_test_split): ] #18 samples

X_train = np.array(X_train_raw, dtype=np.float)
Y_train = np.array(Y_train_raw, dtype=np.float)
X_test = np.array(X_test_raw, dtype=np.float)
Y_test = np.array(Y_test_raw, dtype=np.float)

print(X_train.shape)
print(Y_train.shape) 
print(X_test.shape)
print(Y_test.shape)



