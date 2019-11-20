'''
Description:
Group heart rate records by minutes; 
Then, match and assign label based on sleep stage record.

RR_Record Dimensions: (NUM * MIN * RR); 
    - Number of Patients (NUM); 45
    - Number of Sleep Minutes (MIN); VARIES
    - Number of Interbeat Interval (RR); VARIES

RR_Label Dimensions: (NUM * MIN * 1);
    - Number of Patients (NUM); 45
    - Number of Sleep Minutes (MIN); VARIES
    - Sleep Stage Label (Number between 1-3); 1

1 - Wake
2 - NREM
3 - REM
'''

from os import listdir
from os.path import isfile, join
import pickle

PATH = '/Users/danieltian/Desktop/Start Freshly Code/Sleep Stage Estimation/Sorted Data'

rr_record_total = []
rr_labels_total = []
rr_labels_one_hot_total = []

filenames_raw = [f for f in listdir(PATH) if isfile(join(PATH, f))]
filenames = []
for filename in filenames_raw:
    filename = filename.split('.')[0]
    if (filename not in filenames) and (filename is not ''):
        filenames.append(filename)

for filename in filenames:
    rr_record_patients = []
    rr_labels_patients = []
    rr_labels_one_hot_patients = []

    rr_file = open(PATH + '/' + filename + '.RIT', 'r')
    if rr_file.mode == 'r':
        rr_record =rr_file.read()
        rr_record = rr_record.splitlines()

    time_file = open(PATH + '/' + filename + '.MIN', 'r')
    if time_file.mode == 'r':
        time_record =time_file.read()
        time_record = time_record.splitlines()

    sleep_stage_file = open(PATH + '/' + filename + '.MRL', 'r')
    if sleep_stage_file.mode == 'r':
        sleep_stage_record = sleep_stage_file.read()
        sleep_stage_record = sleep_stage_record.splitlines()


    for i in range(len(time_record)):
        minute, start_idx, end_idx = time_record[i].split(' ')
        rr_record_patients.append(rr_record[int(start_idx)-1: int(end_idx)])

    for i in range(len(time_record)):
        minute, time_start_idx, time_end_idx = time_record[i].split(' ')
        minute = int(minute)
        time_start_idx = int(time_start_idx)
        time_end_idx = int(time_end_idx)

        for j in range(len(sleep_stage_record)):
            hipnogram, sleep_start_idx, sleep_end_idx = sleep_stage_record[j].split(' ')
            sleep_start_idx = int(sleep_start_idx)
            sleep_end_idx = int(sleep_end_idx)
            if (sleep_start_idx <= time_start_idx) and (sleep_end_idx >= time_end_idx):
                sleep_stage = hipnogram[1]
                '''
                if sleep_stage is '1' or sleep_stage is '2':
                    sleep_stage = 2
                    sleep_stage_one_hot = [0, 1, 0, 0]
                elif sleep_stage is '3' or sleep_stage is '4':
                    sleep_stage = 3
                    sleep_stage_one_hot = [0, 0, 1, 0]
                elif sleep_stage is '6' or sleep_stage is '7' or sleep_stage is '8':
                    #combine different wake stages into Wake sleep
                    sleep_stage = 1
                    sleep_stage_one_hot = [1, 0, 0, 0]
                else:
                    #REM sleep stays the same
                    sleep_stage = 4
                    sleep_stage_one_hot = [0, 0, 0, 1]
                '''
                if sleep_stage is '1' or sleep_stage is '2' or sleep_stage is '3' or sleep_stage is '4':
                    sleep_stage = 2
                    sleep_stage_one_hot = [0, 1, 0]
                elif sleep_stage is '6' or sleep_stage is '7' or sleep_stage is '8':
                    #combine different wake stages into Wake sleep
                    sleep_stage = 1
                    sleep_stage_one_hot = [1, 0, 0]
                else:
                    #REM sleep stays the same
                    sleep_stage = 3
                    sleep_stage_one_hot = [0, 0, 1]
                rr_labels_patients.append(int(sleep_stage))
                rr_labels_one_hot_patients.append(sleep_stage_one_hot)
                break

    rr_record_total.append(rr_record_patients)
    rr_labels_total.append(rr_labels_patients)
    rr_labels_one_hot_total.append(rr_labels_one_hot_patients)

print(len(rr_record_total))
print(len(rr_labels_total))
print(len(rr_labels_one_hot_total))

print(len(rr_record_total[1]))
print(len(rr_labels_total[1]))
print(len(rr_labels_one_hot_total[1]))


pickle.dump(rr_record_total, open("rr_record_total.p", "wb"))
pickle.dump(rr_labels_one_hot_total, open("rr_labels_one_hot_total.p", "wb"))
pickle.dump(rr_labels_total, open("rr_labels_total.p", "wb"))









    
    
        
